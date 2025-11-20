import torch
import torch.nn as nn
import pytorch_lightning as pl
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ChordTransformer(pl.LightningModule):
    """Transformer model for chord progression generation."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model components
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src, src_padding_mask=None):
        """Forward pass."""
        # Create causal mask
        seq_len = src.size(1)
        causal_mask = self._generate_square_subsequent_mask(seq_len).to(src.device)

        # If there is a padding mask, set to boolean mask
        if src_padding_mask is not None:
            src_padding_mask = (src_padding_mask == 0)
        
        # Embedding + positional encoding
        x = self.embedding(src) * math.sqrt(self.hparams.d_model)
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(
            x, 
            mask=causal_mask,
            src_key_padding_mask=src_padding_mask
        )
        
        # Output projection
        logits = self.fc_out(x)
        
        return logits
    
    def _generate_square_subsequent_mask(self, sz):
        """
        Generate causal mask for autoregressive training. This is the causal mask
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        src = batch['src']
        tgt = batch['tgt']
        src_mask = batch['src_mask']
        
        # Forward pass
        logits = self(src, src_mask)
        
        # Calculate loss
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=0
        )
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_perplexity', perplexity, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        src = batch['src']
        tgt = batch['tgt']
        src_mask = batch['src_mask']
        
        # Forward pass
        logits = self(src, src_mask)
        
        # Calculate loss
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=0
        )
        
        perplexity = torch.exp(loss)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_perplexity', perplexity, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """test step."""
        src = batch['src']
        tgt = batch['tgt']
        src_mask = batch['src_mask']
        
        # Forward pass
        logits = self(src, src_mask)
        
        # Calculate loss
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=0
        )
        
        perplexity = torch.exp(loss)
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_perplexity', perplexity, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }
    
    def generate(self, vocab, start_tokens=None, max_len=100, temperature=1.0):
        """Generate a chord progression."""
        self.eval()
        
        if start_tokens is None:
            # Start with BOS token
            tokens = [vocab.token2idx['[BOS]']]
        else:
            tokens = start_tokens
            
        with torch.no_grad():
            for _ in range(max_len):
                # Convert to tensor
                src = torch.tensor([tokens]).to(self.device)
                
                # Forward pass
                logits = self(src)
                
                # Get last token logits and apply temperature
                logits = logits[0, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, 1).item()
                
                # Stop if EOS
                if next_token == vocab.token2idx['[EOS]']:
                    break
                    
                tokens.append(next_token)
        
        return vocab.decode(tokens)