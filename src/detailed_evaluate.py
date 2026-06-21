import torch
import torch.nn as nn
from src.dataset import get_data_loaders
from models.model import SimpleCNN

def detailed_evaluate(data_dir, model_path, batch_size=32):
    _, _, test_loader = get_data_loaders(data_dir, batch_size=batch_size)
    
    model = SimpleCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    class_correct = [0, 0]
    class_total = [0, 0]
    classes = ['NORMAL', 'PNEUMONIA']

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(2):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"Fiabilité pour {classes[i]} : {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")

if __name__ == "__main__":
    detailed_evaluate("data/raw", "models/chest_xray_model.pth")
