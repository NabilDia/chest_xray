import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 1. Couches de Convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # 2. Couche de Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # 3. Couches de décision
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # Bloc 1
        x = self.pool(F.relu(self.conv1(x)))
        # Bloc 2
        x = self.pool(F.relu(self.conv2(x)))
        
        # Aplatissement
        x = x.view(-1, 32 * 56 * 56)
        
        # Décision
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    model = SimpleCNN()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Sortie du modèle : {output.shape}")
