import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, features_dim: int, skip_front = 3):
        super(CNN1D, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Input: (batch_size, 1, 64)
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Output: (batch_size, 16, 32)
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Output: (batch_size, 32, 16)
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Output: (batch_size, 64, 8)
        )

        self.skip_front = skip_front
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),  # Output: (batch_size, features_dim - 3)
        )
    
    def forward(self, obs):
        x = obs[:, self.skip_front:]
        # Input shape: (batch_size, 64)
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, 64)
        x = self.conv_layers(x)
        x = x.flatten(1)  # Flatten: (batch_size, 64 * 8)
        x = self.fc_layers(x)
        return torch.cat([obs[:, :self.skip_front], x], dim=1)

class StackedCNN1D(nn.Module):
    """
    A feature extractor that processes stacked observations where the input is flattened.
    Input shape: (batch_size, stack_size * single_obs_dim)
    It reshapes the input, applies CNN1D to each slice, and reshapes the concatenated output.
    Output shape: (batch_size, stack_size * (inner_cnn_features_dim + skip_front))
    """
    def __init__(self, stack_size: int, single_obs_dim: int, inner_cnn_features_dim: int = 16, skip_front: int = 5):
        super(StackedCNN1D, self).__init__()

        # Calculate the final output dimension after concatenation
        cnn_output_dim_per_stack = inner_cnn_features_dim + skip_front

        self.stack_size = stack_size
        self.single_obs_dim = single_obs_dim
        self.skip_front = skip_front
        self.inner_cnn_features_dim = inner_cnn_features_dim
        self.cnn_output_dim_per_stack = cnn_output_dim_per_stack # Store for clarity

        # Instantiate the inner CNN1D
        self.cnn = CNN1D(features_dim=inner_cnn_features_dim, skip_front=skip_front)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        expected_input_dim = self.stack_size * self.single_obs_dim
        assert obs.shape[1] == expected_input_dim, f"Input tensor second dimension {obs.shape[1]} does not match expected {expected_input_dim}"

        # Reshape obs from (batch_size, stack_size * single_obs_dim) to (batch_size * stack_size, single_obs_dim)
        obs_reshaped = obs.reshape(batch_size * self.stack_size, self.single_obs_dim)

        # Apply the inner CNN1D to each (reshaped) observation slice
        # Input shape: (B * B2, D)
        # Output shape: (B * B2, inner_cnn_features_dim + skip_front)
        cnn_features = self.cnn(obs_reshaped)

        # Reshape features back to the final desired output shape
        # Output shape: (B, B2 * (inner_cnn_features_dim + skip_front))
        final_features = cnn_features.reshape(batch_size, -1)
        
        expected_output_dim = self.stack_size * self.cnn_output_dim_per_stack
        assert final_features.shape[1] == expected_output_dim, f"Output tensor second dimension {final_features.shape[1]} does not match expected {expected_output_dim}"


        return final_features
