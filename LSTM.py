from fetch_data import get_data_as_torch
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

class LSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x, mask=None):
        # x shape: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_size)
        
        # Apply fully connected layer to each timestep
        output = self.fc(lstm_out)
        # output shape: (batch, seq_len, input_size)
        
        return output
    

def train(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Now training LSTM on {device}")
    model = model.to(device)
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_Y, batch_mask in train_loader:  # Fixed unpacking order
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            batch_mask = batch_mask.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X, mask=batch_mask)  # Use batch_X directly (already shifted)
            targets = batch_Y  # Use batch_Y directly
            
            mask = batch_mask.unsqueeze(-1)  
            mask = mask.expand_as(targets)  
            
            # Calculate loss only on non-padded tokens
            loss_per_element = criterion(outputs, targets) 
            masked_loss = loss_per_element * mask.float()
            
            # Average over all valid (non-masked) elements
            loss = masked_loss.sum() / mask.sum()
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_Y, batch_mask in val_loader:  # Fixed unpacking order
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)
                batch_mask = batch_mask.to(device)
                
                outputs = model(batch_X, mask=batch_mask)
                targets = batch_Y
                
                mask = batch_mask.unsqueeze(-1).expand_as(targets)
                
                loss_per_element = criterion(outputs, targets)
                masked_loss = loss_per_element * mask.float()
                loss = masked_loss.sum() / mask.sum()
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses


train_loader, val_loader, test_loader = get_data_as_torch(type='degrees',)
model = LSTM()

train_losses, val_losses = train(model=model, train_loader=train_loader, val_loader=val_loader)