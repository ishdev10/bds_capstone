# Streamlit App - Isha Dev

import streamlit as st
import pandas as pd
import os
from PIL import Image
import base64

# Load data
PRED_CSV_PATH = "Pediatric Chest X-ray Pneumonia/labels_and_predictions.csv"
THUMBNAIL_DIR = "Pediatric Chest X-ray Pneumonia/test_images_thumbnails"
GRADCAM_DIR = "Pediatric Chest X-ray Pneumonia/Gradcam images"
CONF_MATRIX_PATH = "Pediatric Chest X-ray Pneumonia/Metrics_Conf_Matrix.png"
ROC_CURVE_PATH = "Pediatric Chest X-ray Pneumonia/ROC_Curve.png"
THRESH_OPTIM_PATH = "Pediatric Chest X-ray Pneumonia/Threshold_Optim.png"
ARCH_FILE = "Pediatric Chest X-ray Pneumonia/pneunet_architecture.py"
ARCH_IMG = "Pediatric Chest X-ray Pneumonia/model_architecture.png"
TRAIN_LOG = "Pediatric Chest X-ray Pneumonia/pneumoniamnist_training_history.csv"
LABELS_PATH = "Pediatric Chest X-ray Pneumonia/labels.csv"

df_preds = pd.read_csv(PRED_CSV_PATH)

# Sidebar
st.sidebar.title("Pediatric Pneumonia Detection \n ## Biomedical Data Design Capstone")
page = st.sidebar.radio("Navigate to", ["Image Explorer", "Dataset Information", "Model Information"])

# Page 1: Image Explorer
if page == "Image Explorer":

    st.header("Explore and Predict")
    st.markdown("Click on a thumbnail below to view full image prediction and Grad-CAM. The images have been rezised and normalized in preparation for classification")

    thumbnails = sorted(os.listdir(THUMBNAIL_DIR))
    selected_img = st.session_state.get("selected_image", None)

    with st.expander("Show image thumbnails to select", expanded=False):
        rows = len(thumbnails) // 6 + 1
        for i in range(rows):
            cols = st.columns(6)
            for j in range(6):
                idx = i * 6 + j
                if idx < len(thumbnails):
                    thumb_file = thumbnails[idx]
                    img_path = os.path.join(THUMBNAIL_DIR, thumb_file)
                    with open(img_path, "rb") as img_file:
                        b64_img = base64.b64encode(img_file.read()).decode()
                        button_html = f"""
                            <button style='background:none;border:none;padding:0;cursor:pointer;' onclick="window.location.reload(false)">
                                <img src="data:image/png;base64,{b64_img}" width="80" style="margin:4px;border-radius:5px;border:2px solid #ccc;" />
                            </button>
                        """
                        if cols[j].button("", key=thumb_file):
                            st.session_state.selected_image = thumb_file
                        cols[j].markdown(button_html, unsafe_allow_html=True)

    selected_img = st.session_state.get("selected_image", None)

    if selected_img:
        full_img_path = os.path.join("Pediatric Chest X-ray Pneumonia/test_images_full", selected_img)
        gradcam_path = os.path.join("Pediatric Chest X-ray Pneumonia/Gradcam images", selected_img.replace("test_img", "test_gradcam"))

        col1, col2 = st.columns(2)
        with col1:
            st.image(full_img_path, caption="Input Image", use_container_width=True)
        with col2:
            st.image(gradcam_path, caption="Grad-CAM", use_container_width=True)

        row = df_preds[df_preds["filename"] == selected_img].iloc[0]
        st.markdown(f"**Prediction:** {row['predicted_label_name']}")
        st.markdown(f"**Confidence:** {row['probability']} (Threshold: {row['threshold_used']})")
        st.markdown(f"**True Label:** {row['true_label_name']}")

# Page 2: Dataset Info
elif page == "Dataset Information":
    st.header("Dataset Information")

    # Dataset description
    st.subheader("Overview")
    st.markdown("""
    This project utilizes a curated dataset of **5,856 pediatric chest X-ray images** labeled as either **"PNEUMONIA"** or **"NORMAL"**. The dataset highlights a critical gap in pediatric imaging: while most existing datasets and diagnostic algorithms focus on adult physiology, pediatric patients present distinct anatomical and pathological features that necessitate dedicated models and data.

    ### Why Pediatric Imaging is Challenging
    - **Patient Compliance**: Young children often struggle to remain still during imaging, leading to motion blur or suboptimal positioning.
    - **Environmental Factors**: The unfamiliar clinical environment—dim lighting, isolation from family, and intimidating equipment—can exacerbate anxiety and reduce image quality.
    - **Physiological Differences**: Pediatric lungs differ significantly from adult lungs in terms of size, structure, and developmental anatomy, making it harder to generalize adult-trained models.

    ### Composition
    - Total images: **5,856**
    - **Normal**: 1,583 images
    - **Pneumonia**: 4,273 images (including both **bacterial** and **viral** cases, though not always separately labeled)
    - Modality: **Anterior–posterior (AP) chest radiographs**
    - Image format: **JPEG**
    - Typical resolution: **~1024×1024 pixels**, but varies slightly

    ### Source & Citation
    This dataset was originally published as part of a landmark study in *Cell* that demonstrated the power of deep learning in medical image diagnostics:

    > **Kermany D, Goldbaum M, Cai W et al.** *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. Cell. 2018; 172(5):1122-1131.  
    > DOI: [10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010)

    **Access the dataset via:**  
    [Kaggle Pediatric Pneumonia Chest X-ray Dataset](https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray)  
    [Mendeley Data DOI: 10.17632/rscbjbr9sj.2](http://dx.doi.org/10.17632/rscbjbr9sj.2)

    ### Splits Used in This Project:
    - **Training Set**: 3,913 images  
    - **Validation Set**: 841 images  
    - **Test Set**: 852 images  

    These splits were designed to maintain label balance and enable robust evaluation.
    """)
    
# Page 3: Model Information
elif page == "Model Information":
    st.header("Model Information")

    st.markdown("""
    The model used in this application, `PneuNet`, is a custom Convolutional Neural Network (CNN) tailored for binary classification of pediatric chest X-ray images as **NORMAL** or **PNEUMONIA**.

    It follows a deep feature extraction and classification design inspired by VGG-like architectures:

    #### Feature Extraction Blocks:
    The model consists of **4 convolutional blocks**, each containing:
    - Two `Conv2d` layers with kernel size 3×3 and padding=1
    - `BatchNorm2d` for stabilization and faster convergence
    - `ReLU` activations to introduce non-linearity
    - `MaxPool2d` layers for downsampling (factor of 2)
    - `Dropout2d` layers to reduce overfitting

    Each block increases the channel depth as follows:
    - Block 1: 64 → 64  
    - Block 2: 128 → 128  
    - Block 3: 256 → 256  
    - Block 4: 512 → 512  

    #### Classification Head:
    - `AdaptiveAvgPool2d` compresses each feature map to a 1×1 output
    - Flattened into a vector
    - Passed through a `Linear` layer with 128 units + `ReLU` + `Dropout`
    - Final `Linear` layer outputs a single logit for binary classification

    #### Summary:
    - **Total Parameters:** ~18.8 million  
    - **Architecture Depth:** 8 Conv layers, 2 Fully Connected layers  
    - **Purpose:** Efficiently extract hierarchical features for pediatric pneumonia detection

    """)

    # Architecture
    st.subheader("Model Architecture")
    with st.expander("Show model architecture code", expanded=False):
        with open(ARCH_FILE, "r") as f:
            code = f.read()
            st.code(code, language="python")
    
    with st.expander("View model diagram", expanded=False):
        st.image(ARCH_IMG, caption="Model Architecture", use_container_width=True)

    # Training history
    st.subheader("Training History")

    # Plot image (loss/accuracy curves)
    st.image("Training_Metrics.png", caption="Training Loss and Accuracy Over Epochs", use_container_width=True)

    # Show all training log entries in dropdown
    with st.expander("See raw training metrics CSV"):
        train_df = pd.read_csv(TRAIN_LOG)
        train_df.insert(0, "Epoch", range(1, len(train_df) + 1))
        train_df.columns = ["Epoch", "Training Loss", "Training Accuracy", "Validation Loss", "Validation Accuracy"]
        with st.container():
            st.dataframe(train_df, hide_index=True, use_container_width=True)

    # Threshold Optimization
    st.subheader("Threshold Optimization")
    st.markdown('<div style="text-align:center;">', unsafe_allow_html=True)
    st.image(THRESH_OPTIM_PATH, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ROC Curve
    st.subheader("ROC Curve")
    st.markdown('<div style="text-align:center;">', unsafe_allow_html=True)
    st.image(ROC_CURVE_PATH, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    st.markdown('<div style="text-align:center;">', unsafe_allow_html=True)
    st.image(CONF_MATRIX_PATH, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
