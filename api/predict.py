import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import json

from models.model import SimpleCNN
from models.enhanced_cnn import EnhancedCNN
from models.resnet_model import ResNet18Classifier

app = FastAPI(
    title="Chest X-Ray Classifier API",
    description="API de classification de radiographies thoraciques (Normal vs Pneumonie)",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Classes
CLASSES = ['NORMAL', 'PNEUMONIA']

# Transformations identiques à l'entraînement
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Chemins des modèles
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATHS = {
    'SimpleCNN': os.path.join(BASE_DIR, 'models', 'chest_xray_model.pth'),
    'EnhancedCNN': os.path.join(BASE_DIR, 'models', 'enhanced_cnn_model.pth'),
    'ResNet18': os.path.join(BASE_DIR, 'models', 'resnet18_model.pth'),
}

# Chargement des modèles
def load_models():
    """Charge tous les modèles disponibles."""
    device = torch.device('cpu')
    loaded = {}
    
    model_classes = {
        'SimpleCNN': SimpleCNN,
        'EnhancedCNN': EnhancedCNN,
        'ResNet18': lambda: ResNet18Classifier(pretrained=False),
    }
    
    for name, model_class in model_classes.items():
        path = MODEL_PATHS[name]
        if os.path.exists(path):
            try:
                model = model_class() if not callable(model_class) or name != 'ResNet18' else model_class()
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
                loaded[name] = model
                print(f"✅ Modèle {name} chargé depuis {path}")
            except Exception as e:
                print(f"❌ Erreur chargement {name}: {e}")
        else:
            print(f"⚠️ Modèle {name} non trouvé à {path}")
    
    return loaded

models_dict = load_models()

def predict_image(image: Image.Image, model, model_name: str):
    """Prédit la classe d'une image avec un modèle donné."""
    image_rgb = image.convert('RGB')
    tensor = transform(image_rgb).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        probs = probabilities[0].tolist()
    
    return {
        'model': model_name,
        'prediction': CLASSES[predicted.item()],
        'confidence': round(confidence.item() * 100, 2),
        'probabilities': {
            'NORMAL': round(probs[0] * 100, 2),
            'PNEUMONIA': round(probs[1] * 100, 2),
        }
    }

@app.get("/")
def root():
    """Endpoint racine."""
    return {
        "message": "Chest X-Ray Classifier API",
        "models_disponibles": list(models_dict.keys()),
        "endpoints": ["/predict", "/predict/all"]
    }

@app.post("/predict/{model_name}")
async def predict_single(model_name: str, file: UploadFile = File(...)):
    """Prédiction avec un seul modèle."""
    if model_name not in models_dict:
        raise HTTPException(status_code=404, detail=f"Modèle '{model_name}' non disponible. Modèles: {list(models_dict.keys())}")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    return predict_image(image, models_dict[model_name], model_name)

@app.post("/predict/all")
async def predict_all(file: UploadFile = File(...)):
    """Prédiction avec tous les modèles disponibles."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    results = []
    for name, model in models_dict.items():
        result = predict_image(image, model, name)
        results.append(result)
    
    return {"predictions": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
