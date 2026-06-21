import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys
import json
import random
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Ajouter le répertoire racine au path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from models.model import SimpleCNN
from models.enhanced_cnn import EnhancedCNN
from models.resnet_model import ResNet18Classifier

# ─────────────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="🩻 Chest X-Ray AI Classifier",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────
# CSS Custom — Thème médical sombre premium
# ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header principal */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d2ff 0%, #7b2ff7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #8892b0;
        font-size: 1.1rem;
        font-weight: 300;
    }
    
    /* Cards de modèle */
    .model-card {
        background: linear-gradient(145deg, rgba(30, 35, 55, 0.9), rgba(20, 25, 40, 0.95));
        border: 1px solid rgba(100, 120, 200, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 210, 255, 0.15);
    }
    .model-card h3 {
        margin: 0 0 0.5rem 0;
        font-weight: 700;
        font-size: 1.2rem;
    }
    
    /* Barre de confiance */
    .confidence-bar-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        overflow: hidden;
        height: 32px;
        margin: 0.5rem 0;
        position: relative;
    }
    .confidence-bar-normal {
        height: 100%;
        border-radius: 12px 0 0 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.85rem;
        color: white;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        float: left;
    }
    .confidence-bar-pneumonia {
        height: 100%;
        border-radius: 0 12px 12px 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.85rem;
        color: white;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        float: right;
    }
    
    /* Prediction badge */
    .prediction-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
    }
    .badge-normal {
        background: linear-gradient(135deg, #00c853, #00e676);
        color: #003d19;
    }
    .badge-pneumonia {
        background: linear-gradient(135deg, #ff1744, #ff5252);
        color: #fff;
    }
    
    /* Metrics card */
    .metric-box {
        background: linear-gradient(145deg, rgba(30, 35, 55, 0.8), rgba(20, 25, 40, 0.9));
        border: 1px solid rgba(100, 120, 200, 0.12);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-box .value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-box .label {
        color: #8892b0;
        font-size: 0.85rem;
        font-weight: 400;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 18, 30, 0.98), rgba(10, 12, 20, 0.99));
    }
    
    /* Upload area */
    .upload-zone {
        border: 2px dashed rgba(0, 210, 255, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: rgba(0, 210, 255, 0.03);
        transition: border-color 0.3s ease;
    }
    .upload-zone:hover {
        border-color: rgba(0, 210, 255, 0.6);
    }
    
    /* Section divider */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 210, 255, 0.3), transparent);
        margin: 2rem 0;
    }
    
    /* Model name colors */
    .simple-cnn-color { color: #64b5f6; }
    .enhanced-cnn-color { color: #ce93d8; }
    .resnet-color { color: #ffd54f; }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Image display */
    .xray-image-container {
        border: 2px solid rgba(0, 210, 255, 0.2);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────
CLASSES = ['NORMAL', 'PNEUMONIA']
MODEL_COLORS = {
    'SimpleCNN': '#64b5f6',
    'EnhancedCNN': '#ce93d8',
    'ResNet18': '#ffd54f',
}
MODEL_ICONS = {
    'SimpleCNN': '🔵',
    'EnhancedCNN': '🟣',
    'ResNet18': '🟡',
}
MODEL_DESCRIPTIONS = {
    'SimpleCNN': 'CNN basique — 2 couches de convolution',
    'EnhancedCNN': 'CNN amélioré — 4 couches + BatchNorm + Dropout',
    'ResNet18': 'ResNet18 — Transfer Learning (ImageNet)',
}

DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ─────────────────────────────────────────────────
# Chargement des modèles (caché en cache)
# ─────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    """Charge tous les modèles disponibles."""
    device = torch.device('cpu')
    loaded = {}
    
    model_configs = {
        'SimpleCNN': {
            'class': SimpleCNN,
            'path': os.path.join(MODELS_DIR, 'chest_xray_model.pth'),
            'args': {},
        },
        'EnhancedCNN': {
            'class': EnhancedCNN,
            'path': os.path.join(MODELS_DIR, 'enhanced_cnn_model.pth'),
            'args': {},
        },
        'ResNet18': {
            'class': ResNet18Classifier,
            'path': os.path.join(MODELS_DIR, 'resnet18_model.pth'),
            'args': {'pretrained': False},
        },
    }
    
    for name, config in model_configs.items():
        if os.path.exists(config['path']):
            try:
                model = config['class'](**config['args'])
                model.load_state_dict(torch.load(config['path'], map_location=device))
                model.eval()
                loaded[name] = model
            except Exception as e:
                st.warning(f"⚠️ Erreur chargement {name}: {e}")
    
    return loaded


@st.cache_data
def load_metrics():
    """Charge les métriques des modèles."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            return json.load(f)
    return None


def predict_image(image: Image.Image, model):
    """Prédit la classe d'une image."""
    image_rgb = image.convert('RGB')
    tensor = transform(image_rgb).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)
        probs = probabilities[0].tolist()
    
    return {
        'NORMAL': round(probs[0] * 100, 2),
        'PNEUMONIA': round(probs[1] * 100, 2),
        'prediction': CLASSES[0] if probs[0] > probs[1] else CLASSES[1],
    }


def get_sample_images():
    """Récupère des images d'exemple depuis le dataset de test."""
    samples = {'NORMAL': [], 'PNEUMONIA': []}
    test_dir = os.path.join(DATA_DIR, 'test')
    
    for cls in CLASSES:
        cls_dir = os.path.join(test_dir, cls)
        if os.path.exists(cls_dir):
            files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            samples[cls] = sorted(files)[:20]  # Max 20 exemples par classe
    
    return samples


# ─────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🩻 Chest X-Ray AI Classifier</h1>
    <p>Comparez les performances de différentes architectures d'IA pour la détection de pneumonie</p>
</div>
<div class="section-divider"></div>
""", unsafe_allow_html=True)

# Chargement des modèles
models_dict = load_all_models()
metrics = load_metrics()

if not models_dict:
    st.error("❌ Aucun modèle trouvé ! Veuillez d'abord entraîner les modèles avec `python src/train_all_models.py`")
    st.stop()

# ─────────────────────────────────────────────────
# SIDEBAR — Dashboard de performance
# ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Dashboard de Performance")
    st.markdown("---")
    
    # Modèles chargés
    st.markdown("### 🤖 Modèles Chargés")
    for name in models_dict:
        icon = MODEL_ICONS.get(name, '🔘')
        st.markdown(f"{icon} **{name}**")
    
    if set(MODEL_ICONS.keys()) - set(models_dict.keys()):
        missing = set(MODEL_ICONS.keys()) - set(models_dict.keys())
        for name in missing:
            st.markdown(f"⚪ ~~{name}~~ *(non entraîné)*")
    
    st.markdown("---")
    
    # Métriques si disponibles
    if metrics:
        st.markdown("### 🏆 Classement par Accuracy")
        
        sorted_models = sorted(metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for rank, (name, m) in enumerate(sorted_models, 1):
            medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(rank, '  ')
            color = MODEL_COLORS.get(name, '#fff')
            st.markdown(f"""
            <div style="
                background: rgba(255,255,255,0.03);
                border-left: 3px solid {color};
                border-radius: 0 8px 8px 0;
                padding: 0.6rem 0.8rem;
                margin: 0.4rem 0;
            ">
                <span style="font-size: 1.1rem;">{medal}</span>
                <span style="color: {color}; font-weight: 600;">{name}</span><br/>
                <span style="font-size: 1.4rem; font-weight: 800; color: white;">{m['accuracy']:.1f}%</span>
                <span style="color: #8892b0; font-size: 0.8rem;"> accuracy</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Métriques détaillées par modèle
        st.markdown("### 📈 Métriques Détaillées")
        selected_model_sidebar = st.selectbox(
            "Sélectionner un modèle",
            list(metrics.keys()),
            key="sidebar_model_select"
        )
        
        if selected_model_sidebar in metrics:
            m = metrics[selected_model_sidebar]
            
            col1, col2 = st.columns(2)
            col1.metric("⏱️ Temps", f"{m.get('training_time_seconds', 'N/A')}s")
            col2.metric("🔢 Params", f"{m.get('total_parameters', 0):,}")
            
            for cls in CLASSES:
                if cls in m.get('per_class', {}):
                    cls_m = m['per_class'][cls]
                    st.markdown(f"**{cls}**")
                    cols = st.columns(3)
                    cols[0].metric("Precision", f"{cls_m['precision']:.1f}%")
                    cols[1].metric("Recall", f"{cls_m['recall']:.1f}%")
                    cols[2].metric("F1", f"{cls_m['f1_score']:.1f}%")
        
        st.markdown("---")
        
        # Matrice de confusion
        st.markdown("### 🔢 Matrice de Confusion")
        cm_model = st.selectbox(
            "Modèle",
            list(metrics.keys()),
            key="cm_model_select"
        )
        
        if cm_model in metrics and 'confusion_matrix' in metrics[cm_model]:
            cm = metrics[cm_model]['confusion_matrix']
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Prédit NORMAL', 'Prédit PNEUMONIA'],
                y=['Réel NORMAL', 'Réel PNEUMONIA'],
                text=[[str(v) for v in row] for row in cm],
                texttemplate="%{text}",
                textfont={"size": 16, "color": "white"},
                colorscale=[[0, '#0d1b2a'], [0.5, '#1b4d8e'], [1, '#00d2ff']],
                showscale=False,
            ))
            fig_cm.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=11),
            )
            st.plotly_chart(fig_cm, use_container_width=True)


# ─────────────────────────────────────────────────
# ZONE PRINCIPALE
# ─────────────────────────────────────────────────

# ─── Sélection de l'image ───
st.markdown("## 🖼️ Sélection de l'Image")

tab_upload, tab_examples = st.tabs(["📤 Upload", "🗂️ Exemples du Dataset"])

image = None
true_label = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Glissez-déposez une radiographie thoracique",
        type=['jpg', 'jpeg', 'png'],
        help="Format accepté : JPG, JPEG, PNG"
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
        true_label = None

with tab_examples:
    samples = get_sample_images()
    
    col_cls1, col_cls2 = st.columns(2)
    
    with col_cls1:
        st.markdown("#### 🫁 Poumon Sain (NORMAL)")
        if samples['NORMAL']:
            selected_normal = st.selectbox(
                "Choisir une image",
                samples['NORMAL'],
                key="normal_select"
            )
            if st.button("📌 Utiliser cette image", key="use_normal"):
                img_path = os.path.join(DATA_DIR, 'test', 'NORMAL', selected_normal)
                image = Image.open(img_path)
                true_label = 'NORMAL'
                st.session_state['selected_image'] = img_path
                st.session_state['true_label'] = 'NORMAL'
    
    with col_cls2:
        st.markdown("#### 🦠 Poumon Pneumonique (PNEUMONIA)")
        if samples['PNEUMONIA']:
            selected_pneumonia = st.selectbox(
                "Choisir une image",
                samples['PNEUMONIA'],
                key="pneumonia_select"
            )
            if st.button("📌 Utiliser cette image", key="use_pneumonia"):
                img_path = os.path.join(DATA_DIR, 'test', 'PNEUMONIA', selected_pneumonia)
                image = Image.open(img_path)
                true_label = 'PNEUMONIA'
                st.session_state['selected_image'] = img_path
                st.session_state['true_label'] = 'PNEUMONIA'
    
    # Bouton image aléatoire
    st.markdown("---")
    col_rand1, col_rand2, col_rand3 = st.columns([1, 2, 1])
    with col_rand2:
        if st.button("🎲 Image Aléatoire", use_container_width=True):
            rand_class = random.choice(CLASSES)
            if samples[rand_class]:
                rand_img = random.choice(samples[rand_class])
                img_path = os.path.join(DATA_DIR, 'test', rand_class, rand_img)
                image = Image.open(img_path)
                true_label = rand_class
                st.session_state['selected_image'] = img_path
                st.session_state['true_label'] = rand_class

# Récupérer l'image de la session si on a cliqué un bouton
if image is None and 'selected_image' in st.session_state:
    image = Image.open(st.session_state['selected_image'])
    true_label = st.session_state.get('true_label')

# ─────────────────────────────────────────────────
# PRÉDICTIONS
# ─────────────────────────────────────────────────
if image is not None:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Affichage image + infos
    col_img, col_results = st.columns([1, 2])
    
    with col_img:
        st.markdown("### 📸 Radiographie")
        st.markdown('<div class="xray-image-container">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if true_label:
            badge_class = 'badge-normal' if true_label == 'NORMAL' else 'badge-pneumonia'
            st.markdown(f"""
            <div style="text-align: center; margin-top: 1rem;">
                <span style="color: #8892b0; font-size: 0.85rem;">Diagnostic réel :</span><br/>
                <span class="prediction-badge {badge_class}">{true_label}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col_results:
        st.markdown("### 🤖 Prédictions des Modèles IA")
        
        results = {}
        for name, model in models_dict.items():
            results[name] = predict_image(image, model)
        
        for name, result in results.items():
            icon = MODEL_ICONS.get(name, '🔘')
            color = MODEL_COLORS.get(name, '#fff')
            desc = MODEL_DESCRIPTIONS.get(name, '')
            
            pred = result['prediction']
            normal_pct = result['NORMAL']
            pneumonia_pct = result['PNEUMONIA']
            
            # Déterminer si la prédiction est correcte
            correctness_html = ""
            if true_label:
                if pred == true_label:
                    correctness_html = '<span style="color: #00e676; font-weight: 600; margin-left: 8px;">✅ Correct</span>'
                else:
                    correctness_html = '<span style="color: #ff5252; font-weight: 600; margin-left: 8px;">❌ Incorrect</span>'
            
            badge_class = 'badge-normal' if pred == 'NORMAL' else 'badge-pneumonia'
            
            # Couleurs des barres
            normal_bar_color = "linear-gradient(90deg, #00c853, #00e676)"
            pneumonia_bar_color = "linear-gradient(90deg, #ff5252, #ff1744)"
            
            st.markdown(f"""
            <div class="model-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h3 style="color: {color};">{icon} {name}</h3>
                        <span style="color: #8892b0; font-size: 0.8rem;">{desc}</span>
                    </div>
                    <div>
                        <span class="prediction-badge {badge_class}">{pred}</span>
                        {correctness_html}
                    </div>
                </div>
                
                <div style="margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="color: #00e676; font-weight: 600; font-size: 0.85rem;">🫁 NORMAL — {normal_pct}%</span>
                        <span style="color: #ff5252; font-weight: 600; font-size: 0.85rem;">🦠 PNEUMONIA — {pneumonia_pct}%</span>
                    </div>
                    <div class="confidence-bar-container">
                        <div class="confidence-bar-normal" style="width: {normal_pct}%; background: {normal_bar_color};">
                            {"" if normal_pct < 15 else f"{normal_pct}%"}
                        </div>
                        <div class="confidence-bar-pneumonia" style="width: {pneumonia_pct}%; background: {pneumonia_bar_color};">
                            {"" if pneumonia_pct < 15 else f"{pneumonia_pct}%"}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # ─── Graphique comparatif ───
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Comparaison Visuelle des Modèles")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Bar chart — Confiance par modèle
        model_names = list(results.keys())
        normal_scores = [results[m]['NORMAL'] for m in model_names]
        pneumonia_scores = [results[m]['PNEUMONIA'] for m in model_names]
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name='NORMAL',
            x=model_names,
            y=normal_scores,
            marker_color='#00e676',
            text=[f"{v:.1f}%" for v in normal_scores],
            textposition='auto',
            textfont=dict(size=14, color='white'),
        ))
        fig_bar.add_trace(go.Bar(
            name='PNEUMONIA',
            x=model_names,
            y=pneumonia_scores,
            marker_color='#ff5252',
            text=[f"{v:.1f}%" for v in pneumonia_scores],
            textposition='auto',
            textfont=dict(size=14, color='white'),
        ))
        fig_bar.update_layout(
            title=dict(text="Confiance par Classe", font=dict(size=16, color='white')),
            barmode='group',
            height=380,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8892b0'),
            legend=dict(orientation="h", y=-0.15, font=dict(color='white')),
            yaxis=dict(range=[0, 100], gridcolor='rgba(255,255,255,0.05)', title="Confiance (%)"),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col_chart2:
        # Radar chart — Force de conviction
        if metrics:
            categories = ['Accuracy', 'Precision\n(NORMAL)', 'Recall\n(NORMAL)', 
                         'Precision\n(PNEUMONIA)', 'Recall\n(PNEUMONIA)']
            
            fig_radar = go.Figure()
            
            for name in results:
                if name in metrics:
                    m = metrics[name]
                    values = [
                        m['accuracy'],
                        m['per_class']['NORMAL']['precision'],
                        m['per_class']['NORMAL']['recall'],
                        m['per_class']['PNEUMONIA']['precision'],
                        m['per_class']['PNEUMONIA']['recall'],
                    ]
                    values.append(values[0])  # Fermer le polygone
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories + [categories[0]],
                        fill='toself',
                        name=name,
                        line=dict(color=MODEL_COLORS.get(name, '#fff')),
                        fillcolor=f"rgba({int(MODEL_COLORS.get(name, '#fff')[1:3], 16)}, "
                                  f"{int(MODEL_COLORS.get(name, '#fff')[3:5], 16)}, "
                                  f"{int(MODEL_COLORS.get(name, '#fff')[5:7], 16)}, 0.1)",
                    ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.1)'),
                    bgcolor='rgba(0,0,0,0)',
                    angularaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                ),
                title=dict(text="Performance Globale (Radar)", font=dict(size=16, color='white')),
                height=380,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#8892b0'),
                legend=dict(orientation="h", y=-0.15, font=dict(color='white')),
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("📊 Les métriques de performance seront disponibles après l'entraînement des modèles.")

    # ─── Tableau récapitulatif ───
    if metrics:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📋 Tableau Récapitulatif")
        
        table_data = []
        for name in results:
            row = {
                'Modèle': f"{MODEL_ICONS.get(name, '')} {name}",
                'Prédiction': results[name]['prediction'],
                'Confiance NORMAL': f"{results[name]['NORMAL']:.1f}%",
                'Confiance PNEUMONIA': f"{results[name]['PNEUMONIA']:.1f}%",
            }
            if name in metrics:
                row['Accuracy (test)'] = f"{metrics[name]['accuracy']:.1f}%"
                row['Paramètres'] = f"{metrics[name].get('total_parameters', 0):,}"
            if true_label:
                row['Correct ?'] = '✅' if results[name]['prediction'] == true_label else '❌'
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

else:
    # État initial — pas d'image sélectionnée
    st.markdown("""
    <div style="
        text-align: center;
        padding: 4rem 2rem;
        background: rgba(0, 210, 255, 0.03);
        border: 2px dashed rgba(0, 210, 255, 0.2);
        border-radius: 20px;
        margin: 2rem 0;
    ">
        <span style="font-size: 4rem;">🩻</span>
        <h3 style="color: #ccd6f6; margin-top: 1rem;">Sélectionnez une radiographie</h3>
        <p style="color: #8892b0;">
            Uploadez une image ou choisissez un exemple dans l'onglet "Exemples du Dataset"<br/>
            pour voir comment chaque modèle d'IA analyse la radiographie.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #4a5568;">
    <p style="font-size: 0.85rem;">
        🩻 <strong>Chest X-Ray AI Classifier</strong> — Projet de Deep Learning<br/>
        <span style="color: #8892b0;">Comparaison d'architectures CNN pour la détection de pneumonie</span>
    </p>
</div>
""", unsafe_allow_html=True)
