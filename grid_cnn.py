import torch.nn as nn

class GridPredictorCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)  # Output: [1, H, W]
        )

    def forward(self, x):
        return self.model(x)
