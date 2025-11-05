🚀 Multi-Disease Chest X-Ray Classifier using Explainable AI (XAI)
This project introduces a state-of-the-art deep learning system designed to automatically classify multiple respiratory diseases from chest X-ray (CXR) images. By integrating Explainable AI (XAI) techniques, it transforms a "black-box" model into a transparent diagnostic assistant, aiding clinicians in making faster, more confident, and accurate decisions.

🧠 Hybrid Model Architecture
The classifier uses a powerful hybrid deep learning architecture to ensure high accuracy and feature diversity.
DenseNet121: Used as a backbone to efficiently spot tiny details and texture shifts in the images.
Swin Transformer: Leveraged to capture the broader, global context and spatial dependencies within the X-ray image.
Convolutional Block Attention Module (CBAM): This attention mechanism is integrated to guide the model to focus only on the most diagnostically significant regions of the lung, enhancing accuracy and clinical relevance.


Key Feature: Explainable AI (XAI)
To build clinical trust, the model's decisions are made transparent using XAI tools:
Grad-CAM (Gradient-weighted Class Activation Mapping): Generates class-specific heatmaps overlaid on the original X-ray, visually highlighting the critical lung regions that led to the model's prediction.
SHAP (SHapley Additive exPlanations): Provides a quantitative view of feature importance, offering a deeper, feature-level understanding of the model's reasoning.

📊 Dataset
The model was trained and evaluated on a comprehensive dataset totaling 16,549 images compiled from public sources, ensuring resilience against noise and out-of-distribution images.
Class,Image Count
Lung Opacity,"6,012"
Non-Chest Images,"4,876"
COVID-19,"3,616"
Viral Pneumonia,"1,345"
Tuberculosis (TB),700
Total Images,"16,549"

Data Preprocessing:
All images undergo a rigorous preprocessing pipeline to ensure consistent quality:
Normalization: Converted to grayscale and resized to a uniform 256x256 pixels.
Contrast Enhancement: Histogram Equalization is applied to enhance the visibility of subtle lung details.
Integrity Check: Duplicate and corrupted files are automatically filtered.

💻 Deployment and Usage
The system is deployed as an accessible Streamlit-based web application.
Real-time Diagnosis: Allows users to upload a chest X-ray image and receive instant prediction results.
Visual Feedback: Presents the predicted disease class, confidence scores, and the Grad-CAM heatmap for visual interpretation.
Clinician-Friendly: Designed with an intuitive interface to require minimal technical expertise, making it suitable for quick, reliable diagnosis in hospitals and remote telemedicine settings.

use command: streamlit run app.py

📈 Evaluation
The model's performance is rigorously evaluated using a comprehensive set of metrics, including Accuracy, Precision, Recall, F1-Score, AUROC (Area Under Receiver Operating Characteristic Curve), and Confusion Matrices to ensure robust performance across all disease classes.

🛣️ Future Work
To further enhance the clinical utility and robustness of the system, future improvements should focus on:
Expanded Dataset: Acquiring more data from diverse hospitals and imaging equipment to improve generalization across different patient demographics and image qualities.
Clinical Integration: Incorporating patient metadata (e.g., age, gender, symptoms) to provide context-aware predictions.
Multi-Label Classification: Enabling the model to identify and report multiple pathologies on a single X-ray image simultaneously.
