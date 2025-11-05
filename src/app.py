import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf

# ========================
# CONFIG
# ========================
DISEASE_MODEL_PATH = r"D:\pjt\src\densenet_best.pth"
DETECTOR_MODEL_PATH = r"D:\pjt\src\best_model_cpu.h5"  # Detector trained on DenseNet features
CLASS_NAMES = ["COVID", "Lung_opacity", "Tuberculosis", "Viral Pneumonia"]
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# LOAD MODELS
# ========================
@st.cache_resource
def load_disease_model():
    model = models.densenet121(pretrained=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(CLASS_NAMES))
    model.load_state_dict(torch.load(DISEASE_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

disease_model = load_disease_model()

@st.cache_resource
def load_detector_model():
    model = tf.keras.models.load_model(DETECTOR_MODEL_PATH)
    return model

detector_model = load_detector_model()

# ========================
# DenseNet feature extractor for detector
# ========================
feature_extractor = models.densenet121(pretrained=True)
feature_extractor.classifier = nn.Identity()  # remove classifier
feature_extractor.to(DEVICE)
feature_extractor.eval()

# ========================
# TRANSFORMS
# ========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========================
# HELPER FUNCTIONS
# ========================
def predict_image(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = disease_model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, preds = torch.max(probs, 1)
    return preds.item(), confidence.item(), probs[0].cpu().numpy(), img_tensor

def generate_gradcam(image_tensor, class_idx):
    gradients, activations = [], []

    def save_gradients(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    def save_activations(module, input, output):
        activations.append(output)

    last_conv_layer = disease_model.features[-1]
    hook_a = last_conv_layer.register_forward_hook(save_activations)
    hook_g = last_conv_layer.register_backward_hook(save_gradients)

    disease_model.zero_grad()
    output = disease_model(image_tensor)
    pred_class = output[:, class_idx]
    pred_class.backward()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1,2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i,:,:]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    hook_a.remove()
    hook_g.remove()
    return cam

def overlay_gradcam(img: np.ndarray, cam: np.ndarray, threshold: float = 0.4):
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam_resized = cam_resized / cam_resized.max()
    mask = np.uint8(cam_resized > threshold)
    heatmap = cv2.applyColorMap(np.uint8(255*cam_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)/255
    highlighted = np.float32(img)/255
    highlighted = highlighted*(1-mask[:,:,None]) + heatmap*mask[:,:,None]
    highlighted = highlighted / np.max(highlighted)
    return np.uint8(255*highlighted)

def estimate_severity(cam: np.ndarray, threshold: float = 0.4):
    total_pixels = cam.size
    activated_pixels = np.sum(cam > threshold)
    coverage = activated_pixels / total_pixels
    if coverage < 0.2:
        return "Mild"
    elif coverage < 0.5:
        return "Moderate"
    else:
        return "Severe"

def get_safe_recommendation(disease: str, severity: str):
    messages = {
        "COVID": {"Mild":"Possible early-stage COVID signs. Consult a doctor and follow isolation guidelines.",
                  "Moderate":"Signs suggest moderate COVID involvement. Seek medical evaluation.",
                  "Severe":"High likelihood of severe COVID signs. Urgent medical attention recommended."},
        "Lung_opacity": {"Mild":"Minor lung opacity detected. Follow-up with a doctor advised.",
                         "Moderate":"Moderate lung involvement detected. Consult a healthcare provider soon.",
                         "Severe":"Extensive lung involvement detected. Urgent medical evaluation recommended."},
        "Tuberculosis": {"Mild":"Possible early-stage TB signs. Please consult a doctor for testing.",
                         "Moderate":"Moderate TB signs detected. Medical evaluation recommended.",
                         "Severe":"Advanced TB signs detected. Immediate medical attention required."},
        "Viral Pneumonia": {"Mild":"Mild pneumonia signs detected. Monitor symptoms and consult a doctor.",
                            "Moderate":"Moderate pneumonia signs detected. Seek medical evaluation.",
                            "Severe":"Severe pneumonia signs detected. Urgent medical attention recommended."}
    }
    return messages.get(disease, {}).get(severity, "Please consult a healthcare professional.")

# ========================
# Chest detector preprocessing & prediction (Fixed)
# ========================
def preprocess_detector(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = feature_extractor(img_tensor)
        features = torch.flatten(features, 1)
    return features.cpu().numpy()

def is_chest_xray(image: Image.Image, threshold=0.5, invert=True, debug=False):
    features_np = preprocess_detector(image)
    features_tf = tf.convert_to_tensor(features_np, dtype=tf.float32)
    
    prob = float(detector_model.predict(features_tf)[0][0])
    
    # Invert if detector was trained with swapped labels
    if invert:
        prob = 1.0 - prob

    if debug:
        st.write(f"Detector raw probability: {prob:.4f}")
    
    return prob > threshold, prob

# ========================
# STREAMLIT UI
# ========================
st.set_page_config(page_title="Chest X-Ray Disease Classifier", layout="centered")
st.title("🩺 Chest X-Ray Disease Classifier")
st.write("Upload a chest X-ray and the model will predict the disease with explainability (Grad-CAM) and severity estimation.")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    try:
        # Set invert=True if detector labels were swapped during training
        is_chest, chest_conf = is_chest_xray(image, threshold=0.5, invert=True, debug=True)
    except Exception as e:
        st.error(f"Error in chest detection: {e}")
        is_chest, chest_conf = False, 0.0

    if not is_chest:
        st.error(f"⚠ This image does not look like a valid chest X-ray (Confidence: {chest_conf:.2f})")
    else:
        pred_idx, confidence, probs, img_tensor = predict_image(image)

        st.subheader("Prediction Results")
        st.write(f"Predicted Disease: *{CLASS_NAMES[pred_idx]}*")
        st.write(f"Confidence: *{confidence*100:.2f}%*")
        st.bar_chart({CLASS_NAMES[i]: probs[i] for i in range(len(CLASS_NAMES))})

        cam = generate_gradcam(img_tensor, pred_idx)
        img_resized = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
        gradcam_result = overlay_gradcam(img_resized, cam)
        st.subheader("🔍 Explainability (Grad-CAM)")
        st.image(gradcam_result, caption="Highlighted regions show model's focus", use_column_width=True)

        severity = estimate_severity(cam, threshold=0.4)
        recommendation = get_safe_recommendation(CLASS_NAMES[pred_idx], severity)
        st.subheader("⚠ Disease Severity & Recommendation")
        st.write(f"Estimated Severity: *{severity}*")
        st.write(f"Recommendation: *{recommendation}*")