import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Model(nn.Module):
    def __init__(self, n_f = 257, T = 101, num_classes=10):
        super(CNN_Model, self).__init__()
        
        # --- BLOCK 1 ---
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # --- BLOCK 2 ---
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # --- BLOCK 3 ---
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # --- DYNAMIC FLATTEN CALCULATION ---
        self.flatten_size = self._get_flatten_size((2, n_f, T))
        
        # --- DENSE LAYERS ---
        self.fc1 = nn.Linear(in_features=self.flatten_size, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def _get_flatten_size(self, shape):
        """Passes a dummy tensor to calculate the linear input size."""
        x = torch.zeros(1, *shape)
        
        # We manually apply the spatial downsampling steps to the dummy tensor
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = self.conv3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        return x.numel()

    def forward(self, x):
        # 1. Format input: (B, Freq, Time, Channels) -> (B, Channels, Freq, Time)
        if x.dim() == 4 and x.shape[-1] == 2:
            x = x.permute(0, 3, 1, 2)
            
        # --- BLOCK 1 ---
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # --- BLOCK 2 ---
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # --- BLOCK 3 ---
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # --- FLATTEN ---
        x = torch.flatten(x, start_dim=1)
        
        # --- DENSE LAYERS ---
        x = self.fc1(x)
        x = F.relu(x)
        
        # Note: When using F.dropout, you MUST pass `training=self.training`. 
        # This ensures dropout is turned on during training but turned off during model.eval()
        x = F.dropout(x, p=0.5, training=self.training)
        
        logits = self.fc2(x)
        
        return logits