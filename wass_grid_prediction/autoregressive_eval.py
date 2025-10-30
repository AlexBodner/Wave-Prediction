import torch
import numpy as np
from grid_dataloader import get_dataloaders, GridDataset
from train_cnn_delta import BasicGridCNN
import matplotlib.pyplot as plt

def autoregressive_predict(model, initial_context, steps, device):
    """
    initial_context: (context_len, N, M) numpy array
    steps: number of frames to predict
    Returns: (steps, N, M) numpy array of predictions
    """
    context = initial_context.copy()
    preds = []
    for _ in range(steps):
        inp = torch.from_numpy(context).float().unsqueeze(0)  # (1, context_len, N, M)
        # If input has shape (1, 1, context_len, N, M), squeeze the 2nd dim
        if inp.ndim == 5 and inp.shape[1] == 1:
            inp = inp.squeeze(1)
        inp = inp.to(device)
        out = model(inp).cpu().detach().numpy()
        # Ensure output is (N, M)
        if out.ndim == 4:
            out = out.squeeze(0).squeeze(0)
        elif out.ndim == 3:
            out = out.squeeze(0)
        preds.append(out)
        # Slide window: drop oldest, append new pred
        context = np.concatenate([context[1:], out[None, ...]], axis=0)
    return np.stack(preds, axis=0)

def eval_autoregressive(split='valid', n_sequences=5, context_len=3, delta=15, checkpoint_path='best_grid_cnn_delta15.pth'):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    nc_file = "Surfaces_20150305_103500.nc"
    # Use batch_size=1 for easy sequential access
    train_loader, valid_loader = get_dataloaders(nc_file, batch_size=1, fraction=1.0, context_len=context_len, delta=delta)
    loader = train_loader if split == 'train' else valid_loader
    model = BasicGridCNN(in_channels=context_len, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    maes = []
    mses = []
    loader_iter = iter(loader)
    for batch_idx in range(len(loader)):
        try:
            inputs, targets = next(loader_iter)
        except StopIteration:
            break
        # inputs: (1, context_len, N, M), targets: (1, N, M)
        # Build a sequence: get the next N frames after the context window
        context = inputs.cpu().numpy()[0]
        # Predict as many steps as possible until the end of the dataset
        steps = 30
        gt_seq = []
        # Collect the next 'steps' targets from the loader
        for _ in range(steps):
            try:
                _, next_target = next(loader_iter)
                gt_seq.append(next_target.cpu().numpy()[0])
            except StopIteration:
                break
        gt_seq = np.stack(gt_seq, axis=0) if gt_seq else np.zeros_like(context[0:1])
        # Autoregressive prediction
        preds = autoregressive_predict(model, context, gt_seq.shape[0], device)
        # Compute errors
        mae = np.mean(np.abs(preds - gt_seq))
        mse = np.mean((preds - gt_seq) ** 2)
        maes.append(mae)
        mses.append(mse)
        print(f"Sequence {batch_idx}: MAE={mae:.4f}, MSE={mse:.4f}")
        # Optionally plot first sequence
        if batch_idx == 0 or batch_idx < n_sequences:
            plt.figure(figsize=(10, 4))
            plt.plot(np.mean(np.abs(preds - gt_seq), axis=(1,2)), label='MAE per step')
            plt.title(f'Autoregressive MAE per step ({split})')
            plt.xlabel('Step')
            plt.ylabel('MAE')
            plt.legend()
            plt.show()

            # Visualize predictions vs truth at selected steps
            steps_to_plot = [0, 1, 2, 3, 4]  
            n_plot = len(steps_to_plot)
            fig, axs = plt.subplots(2, n_plot, figsize=(3*n_plot, 6))
            for i, step in enumerate(steps_to_plot):
                if step >= preds.shape[0]:
                    continue
                axs[0, i].imshow(gt_seq[step], cmap='viridis')
                axs[0, i].set_title(f'Ground truth step {step+1}')
                axs[0, i].axis('off')
                axs[1, i].imshow(preds[step], cmap='viridis')
                axs[1, i].set_title(f'Pred step {step+1}')
                axs[1, i].axis('off')
            fig.suptitle(f'Prediccion Autoregresiva ({split} set')
            plt.tight_layout()
            plt.show()
    print(f"Average {split} MAE: {np.mean(maes):.4f}, MSE: {np.mean(mses):.4f}")

if __name__ == "__main__":
    print("Autoregressive evaluation on validation set...")
    eval_autoregressive(split='valid', n_sequences=5, context_len=3, delta=15, checkpoint_path='best_grid_cnn_delta15_fraction1.pth')
    print("Autoregressive evaluation on training set...")
    eval_autoregressive(split='train', n_sequences=5, context_len=3, delta=15)
