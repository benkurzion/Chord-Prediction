import torch
import torch.nn as nn
import pytorch_lightning as pl
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
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
        """
        Standard forward pass: adds PE based on the token's position in the sequence.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TypePositionalEncoding(PositionalEncoding):
    """
    Positional encoding strategy that assigns a PE vector based on the
    token's identity (type), not its current sequence position.
    
    A token's position is fixed to the *first time* it was encountered.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__(d_model, max_len, dropout)
        self.token_to_pos_index = {}
        self.next_pos_index = 0
        
    def forward(self, src_token_ids, embedded_x):
        """
        Custom forward pass that uses a token's *identity* to look up a fixed PE index.
        
        :param src_token_ids: The tensor of token IDs (e.g., from the input batch 'src'). Shape: (B, L)
        :param embedded_x: The already embedded tensor (input to PE). Shape: (B, L, D)
        :return: The embedded tensor with Type-based Positional Encoding added.
        """
        B, L, D = embedded_x.size()

        type_pe = torch.zeros(B, L, D, device=embedded_x.device)

        if B > 1:
            print("Warning: TypePositionalEncoding is designed for B=1 (e.g., generation) or requires state management for batched training.")
        if B == 1:
            src_tokens = src_token_ids[0].tolist() # Get the single sequence of tokens
            
            for i in range(L):
                token_id = src_tokens[i]
                if token_id not in self.token_to_pos_index:
                    if self.next_pos_index < self.pe.size(1):
                        self.token_to_pos_index[token_id] = self.next_pos_index
                        self.next_pos_index += 1
                    else:
                        # Handle overflow (e.g., assign max_len - 1's PE)
                        self.token_to_pos_index[token_id] = self.pe.size(1) - 1

                pos_index = self.token_to_pos_index[token_id]
                type_pe[0, i, :] = self.pe[0, pos_index, :]
                
        else: 
            return super().forward(embedded_x)
              
        # Add the type-based PE and apply dropout
        x = embedded_x + type_pe
        return self.dropout(x)
        
    def reset_state(self):
        """
        Call this before processing a new, independent sequence (like a new sample 
        in a batch or a new generation session).
        """
        self.token_to_pos_index = {}
        self.next_pos_index = 0

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
        pe_strategy: str = 'sinusoidal'
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model components
        self.embedding = nn.Embedding(vocab_size, d_model)
        if pe_strategy == 'type':
            self.pos_encoder = TypePositionalEncoding(d_model, max_seq_len, dropout)
        elif pe_strategy == 'sinusoidal':
            self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        else:
            raise ValueError(f"Unknown positional encoding strategy: {pe_strategy}")
        
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
        if isinstance(self.pos_encoder, TypePositionalEncoding):
            x = self.pos_encoder(src, x) 
        else:
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

        if isinstance(self.pos_encoder, TypePositionalEncoding):
            if batch['src'].size(0) == 1:
                self.pos_encoder.reset_state()
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
        if isinstance(self.pos_encoder, TypePositionalEncoding):
            if batch['src'].size(0) == 1:
                self.pos_encoder.reset_state()

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
        if isinstance(self.pos_encoder, TypePositionalEncoding):
            if batch['src'].size(0) == 1:
                self.pos_encoder.reset_state()

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

        if isinstance(self.pos_encoder, TypePositionalEncoding):
            self.pos_encoder.reset_state()
        
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

    def generate_genre(self, vocab, start_tokens=None, genre: str = None, max_len=100, temperature=1.0):
        """
        Generate a chord progression conditioned on an optional genre.

        Args:
            vocab: ChordVocabulary instance
            start_tokens: Optional list of token IDs to start generation (without genre/BOS)
            genre: Optional genre string (e.g., "pop") to condition generation
            max_len: Maximum sequence length
            temperature: Sampling temperature

        Returns:
            Generated chord sequence as a string
        """
        self.eval()

        # Reset TypePositionalEncoding state if used
        if isinstance(self.pos_encoder, TypePositionalEncoding):
            self.pos_encoder.reset_state()

        # Start with genre token + BOS
        tokens = []
        if genre is not None:
            genre_token = vocab.token2idx.get(f"[GENRE_{genre.upper()}]", vocab.token2idx["[GENRE_UNKNOWN]"])
            tokens.append(genre_token)
        tokens.append(vocab.token2idx['[BOS]'])

        # Append any additional start tokens
        if start_tokens is not None:
            tokens += start_tokens

        with torch.no_grad():
            for _ in range(max_len):
                src = torch.tensor([tokens], device=self.device)
                logits = self(src)  # forward pass

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