import torch
import torch.nn as nn
from torchvision import models

class ResNet18Classifier(nn.Module):
    """ResNet18 pré-entraîné adapté pour la classification binaire de radiographies."""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet18Classifier, self).__init__()
        
        # Charger le ResNet18 pré-entraîné sur ImageNet
        if pretrained:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.resnet = models.resnet18(weights=None)
        
        # Remplacer la dernière couche FC pour 2 classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

if __name__ == "__main__":
    model = ResNet18Classifier()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Sortie du modèle ResNet18 : {output.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre total de paramètres : {total_params:,}")
