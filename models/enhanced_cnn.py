import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        
        # Bloc 1: 3 -> 32 canaux
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Bloc 2: 32 -> 64 canaux
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Bloc 3: 64 -> 128 canaux
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Bloc 4: 128 -> 256 canaux
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Couches fully-connected
        # After 4 pools: 224 / 2^4 = 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        
        # Dropout pour la régularisation
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Bloc 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Bloc 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Bloc 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Bloc 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Aplatissement
        x = x.view(-1, 256 * 14 * 14)
        
        # Couches de décision avec dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

if __name__ == "__main__":
    model = EnhancedCNN()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Sortie du modèle EnhancedCNN : {output.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre total de paramètres : {total_params:,}")
