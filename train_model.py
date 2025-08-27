import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from grid_dataset import GridSequenceDataset
from grid_cnn import GridPredictorCNN
from build_grid_dataset_from_files import build_dataset

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_paths = build_dataset()
    train_dataset = GridSequenceDataset(dataset_paths)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = GridPredictorCNN(in_channels=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            print(inputs)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "grid_predictor.pth")
    print("✅ Modelo entrenado y guardado.")

if __name__ == "__main__":
    train()
