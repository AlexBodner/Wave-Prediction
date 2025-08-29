import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from grid_dataset import GridSequenceDataset
from grid_cnn import GridPredictorCNN
from build_grid_dataset_from_files import build_dataset

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_paths = build_dataset()
    n = len(dataset_paths)
    train_end = int(n * 0.75)
    valid_end = int(n * 0.90)

    train_paths = dataset_paths[:train_end]
    valid_paths = dataset_paths[train_end:valid_end]
    test_paths = dataset_paths[valid_end:]

    train_dataset = GridSequenceDataset(train_paths)
    valid_dataset = GridSequenceDataset(valid_paths)
    test_dataset = GridSequenceDataset(test_paths)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = GridPredictorCNN(in_channels=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            inputs = torch.nan_to_num(inputs, nan=0.0)
            targets = torch.nan_to_num(targets, nan=0.0)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                inputs = torch.nan_to_num(inputs, nan=0.0)
                targets = torch.nan_to_num(targets, nan=0.0)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(valid_loader)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), "grid_predictor.pth")
    print("✅ Modelo entrenado y guardado.")

if __name__ == "__main__":
    train()