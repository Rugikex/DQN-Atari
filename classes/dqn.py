import torch
import torch.nn as nn


class DeepQNetwork(nn.Module):
    """
    Deep Q Network

    Parameters
    ----------
    num_actions: int
        Number of actions
    input_shape: tuple
        Shape of the input
        default: (4, 84, 84)
    """

    def __init__(self, num_actions: int, input_shape: tuple = (4, 84, 84)):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(self.cnn(torch.zeros(1, *input_shape)).numel(), 512)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
