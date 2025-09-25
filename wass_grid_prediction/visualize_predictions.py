import torch
import matplotlib.pyplot as plt
import numpy as np
from grid_dataloader import get_dataloaders
from train_cnn import BasicGridCNN
from mpl_toolkits.mplot3d import Axes3D


# 2D Visualization function
def plot_pred_vs_actual(input_frames, pred, actual, idx=None, save_path=None):
    fig, axs = plt.subplots(1, 5, figsize=(18, 4))
    for i in range(3):
        axs[i].imshow(input_frames[i], cmap='viridis')
        axs[i].set_title(f'Input t-{2-i}')
        axs[i].axis('off')
    im3 = axs[3].imshow(actual, cmap='viridis')
    axs[3].set_title('Actual')
    axs[3].axis('off')
    im4 = axs[4].imshow(pred, cmap='viridis')
    axs[4].set_title('Predicted')
    axs[4].axis('off')
    fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)
    fig.colorbar(im4, ax=axs[4], fraction=0.046, pad=0.04)
    if idx is not None:
        fig.suptitle(f'Sample {idx}')
    if save_path:
        plt.savefig(save_path)
    plt.show()

# 3D Visualization function
def plot_3d_surface(grid, title="3D Surface", elev=30, azim=45, cmap='viridis'):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    N, M = grid.shape
    X, Y = np.meshgrid(np.arange(M), np.arange(N))
    surf = ax.plot_surface(X, Y, grid, cmap=cmap, edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Z Value')
    plt.tight_layout()
    plt.show()

def visualize_predictions(n_samples=5, batch_size=1, split='valid', checkpoint_path='best_grid_cnn.pth'):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    nc_file = "Surfaces_20150305_103500.nc"
    # Use batch_size=1 for easy visualization
    train_loader, valid_loader = get_dataloaders(nc_file, batch_size=batch_size, fraction=1.0)
    loader = train_loader if split == 'train' else valid_loader
    model = BasicGridCNN(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    count = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs).squeeze(1).cpu().numpy()  # (B, N, M)
            inputs_np = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            for i in range(inputs_np.shape[0]):
                idx = batch_idx*batch_size+i
                plot_pred_vs_actual(inputs_np[i], outputs[i], targets_np[i], idx=idx)
                # 3D plot for actual and predicted
                plot_3d_surface(targets_np[i], title=f"Actual 3D Surface (Sample {idx})")
                plot_3d_surface(outputs[i], title=f"Predicted 3D Surface (Sample {idx})")
                # 3D plot for absolute difference
                abs_diff = np.abs(targets_np[i] - outputs[i])
                plot_3d_surface(abs_diff, title=f"Absolute Error 3D Surface (Sample {idx})", cmap='inferno')
                count += 1
                if count >= n_samples:
                    return

if __name__ == "__main__":
    print("Visualizing predictions on validation set...")
    visualize_predictions(n_samples=5, split='valid')
    print("Visualizing predictions on training set...")
    visualize_predictions(n_samples=5, split='train')
