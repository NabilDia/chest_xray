import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms():
    """
    Définit les transformations à appliquer aux images.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

def get_data_loaders(data_dir, batch_size=32):
    """
    Crée les chargeurs de données.
    """
    transform = get_transforms()
    
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    data_path = "data/raw"
    train_loader, val_loader, test_loader = get_data_loaders(data_path)
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
