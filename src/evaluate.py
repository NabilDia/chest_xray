import torch
from src.dataset import get_data_loaders
from models.model import SimpleCNN

def evaluate_model(data_dir, model_path, batch_size=32):
    # 1. Charger les données de test
    _, _, test_loader = get_data_loaders(data_dir, batch_size=batch_size)
    
    # 2. Reconstruire l'architecture
    model = SimpleCNN()
    
    # 3. Charger les connaissances (poids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Mode évaluation
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"Précision sur le jeu de test : {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    evaluate_model("data/raw", "models/chest_xray_model.pth")
