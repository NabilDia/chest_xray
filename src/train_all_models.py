import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from src.dataset import get_data_loaders
from models.model import SimpleCNN
from models.enhanced_cnn import EnhancedCNN
from models.resnet_model import ResNet18Classifier


def evaluate_model(model, test_loader, device):
    """Évalue un modèle et retourne les métriques détaillées."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    classes = ['NORMAL', 'PNEUMONIA']
    accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'accuracy': round(accuracy, 2),
        'per_class': {}
    }
    
    for i, cls in enumerate(classes):
        metrics['per_class'][cls] = {
            'precision': round(float(precision[i]) * 100, 2),
            'recall': round(float(recall[i]) * 100, 2),
            'f1_score': round(float(f1[i]) * 100, 2),
        }
    
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def train_single_model(model, model_name, train_loader, val_loader, test_loader, 
                        device, epochs=3, learning_rate=0.001):
    """Entraîne un seul modèle et retourne les métriques."""
    print(f"\n{'='*60}")
    print(f"🧠 Entraînement du modèle : {model_name}")
    print(f"{'='*60}")
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 10 == 0:
                print(f"  Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"  📊 Époque {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
    
    training_time = time.time() - start_time
    print(f"  ⏱️ Temps d'entraînement : {training_time:.1f}s")
    
    # Sauvegarde du modèle
    save_paths = {
        'SimpleCNN': 'models/chest_xray_model.pth',
        'EnhancedCNN': 'models/enhanced_cnn_model.pth',
        'ResNet18': 'models/resnet18_model.pth',
    }
    save_path = save_paths.get(model_name, f'models/{model_name.lower()}_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"  💾 Modèle sauvegardé : {save_path}")
    
    # Évaluation
    print(f"  🔍 Évaluation sur le jeu de test...")
    metrics = evaluate_model(model, test_loader, device)
    metrics['training_time_seconds'] = round(training_time, 1)
    metrics['epochs'] = epochs
    
    total_params = sum(p.numel() for p in model.parameters())
    metrics['total_parameters'] = total_params
    
    print(f"  ✅ Accuracy: {metrics['accuracy']:.2f}%")
    for cls, cls_metrics in metrics['per_class'].items():
        print(f"     {cls}: Precision={cls_metrics['precision']:.1f}% | Recall={cls_metrics['recall']:.1f}% | F1={cls_metrics['f1_score']:.1f}%")
    
    return metrics


def train_all_models(data_dir="data/raw", epochs=3, batch_size=32):
    """Entraîne tous les modèles et sauvegarde les métriques."""
    print("🩻 Chest X-Ray Classifier — Entraînement Multi-Modèles")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device: {device}")
    
    # Chargement des données
    print("📂 Chargement des données...")
    train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size=batch_size)
    print(f"   Train: {len(train_loader.dataset)} images")
    print(f"   Val: {len(val_loader.dataset)} images")
    print(f"   Test: {len(test_loader.dataset)} images")
    
    # Définition des modèles à entraîner
    models_to_train = {
        'SimpleCNN': SimpleCNN(),
        'EnhancedCNN': EnhancedCNN(),
        'ResNet18': ResNet18Classifier(pretrained=True),
    }
    
    all_metrics = {}
    
    for model_name, model in models_to_train.items():
        metrics = train_single_model(
            model, model_name, train_loader, val_loader, test_loader,
            device, epochs=epochs
        )
        all_metrics[model_name] = metrics
    
    # Sauvegarde des métriques globales
    metrics_path = "models/metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n📊 Métriques sauvegardées dans : {metrics_path}")
    
    # Résumé final
    print(f"\n{'='*60}")
    print("🏆 RÉSUMÉ FINAL")
    print(f"{'='*60}")
    for name, m in sorted(all_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"  {name:15s} | Accuracy: {m['accuracy']:6.2f}% | Params: {m['total_parameters']:>12,}")
    
    return all_metrics


if __name__ == "__main__":
    train_all_models("data/raw", epochs=3)
