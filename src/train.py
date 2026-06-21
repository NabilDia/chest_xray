import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_data_loaders
from models.model import SimpleCNN


def train_model(data_dir, epochs=5, batch_size=32, learning_rate=0.001):
    train_loader, val_loader, _ = get_data_loaders(data_dir, batch_size=batch_size)
    model = SimpleCNN()
    
    # La "règle" pour mesurer l'erreur
    criterion = nn.CrossEntropyLoss()
    # Le "moteur" qui ajuste les paramètres du modèle
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()   # 1. Reset
            outputs = model(images)  # 2. Prédiction
            loss = criterion(outputs, labels) # 3. Erreur
            loss.backward()         # 4. Analyse de l'erreur
            optimizer.step()        # 5. Correction
            
            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Batch {i+1}/{len(train_loader)} - Loss actuelle: {loss.item():.4f}")
            
        print(f"Époque {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")
    
    # Sauvegarde du modèle
    save_path = "models/chest_xray_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Modèle sauvegardé sous : {save_path}")
    
    return model

if __name__ == "__main__":
    train_model("data/raw", epochs=2)
