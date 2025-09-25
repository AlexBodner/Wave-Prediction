import torch
from torch import nn, optim
from grid_dataloader import get_dataloaders
from tqdm import tqdm

class BasicGridCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
    def forward(self, x):
        return self.net(x)  # (B, N, M)

def train( batch_size = 64,
        epochs = 20,
        fraction = 0.1,  # Use 5% of the data 
    ):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc_file = "Surfaces_20150305_103500.nc" 
    train_loader, valid_loader = get_dataloaders(nc_file, batch_size=batch_size,fraction=0.2)
    model = BasicGridCNN(in_channels=3, out_channels=1).to(device)
    criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_valid_mae = float('inf')
    best_model_state = None
    print ("🚀 Starting training with parameters: "
    f"Epochs: {epochs}, Batch Size: {batch_size}, Fraction of Data: {fraction*100}%,")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for inputs, targets in train_iter:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)  # (B, 1, N, M)
            optimizer.zero_grad()
            outputs = model(inputs)  # (B, 1, N, M)
            loss = criterion(outputs, targets)
            mae = mae_criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_mae += mae.item()
            train_iter.set_postfix(loss=loss.item(), mae=mae.item())
        avg_train_loss = train_loss / len(train_loader)
        avg_train_mae = train_mae / len(train_loader)

        # Validation
        model.eval()
        valid_loss = 0.0
        valid_mae = 0.0
        valid_iter = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]", leave=False)
        with torch.no_grad():
            for inputs, targets in valid_iter:
                inputs = inputs.to(device)
                targets = targets.to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                mae = mae_criterion(outputs, targets)
                valid_loss += loss.item()
                valid_mae += mae.item()
                valid_iter.set_postfix(loss=loss.item(), mae=mae.item())
        avg_valid_loss = valid_loss / len(valid_loader)
        avg_valid_mae = valid_mae / len(valid_loader)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Train MAE: {avg_train_mae:.4f} | Valid Loss: {avg_valid_loss:.4f} | Valid MAE: {avg_valid_mae:.4f}")

        # Save best model by MAE
        if avg_valid_mae < best_valid_mae:
            best_valid_mae = avg_valid_mae
            best_model_state = model.state_dict()

    if best_model_state is not None:
        torch.save(best_model_state, "best_grid_cnn.pth")
        print(f"✅ Best model saved with Valid MAE: {best_valid_mae:.4f}")
    else:
        print("No model was saved.")

if __name__ == "__main__":
    train()
