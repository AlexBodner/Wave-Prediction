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
        return self.net(x).squeeze(1)  # (B, N, M)

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc_file = "Surfaces_20150305_103500.nc"  # Change if needed
    train_loader, valid_loader = get_dataloaders(nc_file, batch_size=64)
    model = BasicGridCNN(in_channels=3, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 20
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for inputs, targets in train_iter:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)  # (B, 1, N, M)
            optimizer.zero_grad()
            outputs = model(inputs)  # (B, N, M)
            loss = criterion(outputs, targets.squeeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_iter.set_postfix(loss=loss.item())
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        valid_loss = 0.0
        valid_iter = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]", leave=False)
        with torch.no_grad():
            for inputs, targets in valid_iter:
                inputs = inputs.to(device)
                targets = targets.to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets.squeeze(1))
                valid_loss += loss.item()
                valid_iter.set_postfix(loss=loss.item())
        avg_valid_loss = valid_loss / len(valid_loader)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")

    torch.save(model.state_dict(), "unet_grid_predictor.pth")
    print("✅ Model trained and saved.")

if __name__ == "__main__":
    train()
