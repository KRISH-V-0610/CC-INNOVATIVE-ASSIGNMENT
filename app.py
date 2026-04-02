import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from model import EEG_CNN_RNN

st.set_page_config(page_title="Cloud EEG Architecture", layout="wide")

# Load Data
@st.cache_resource
def load_assets():
    model = EEG_CNN_RNN(num_channels=64, num_classes=2)
    model.load_state_dict(torch.load("eeg_model.pth", weights_only=True))
    model.eval()
    eval_data = torch.load("eval_data.pth", weights_only=False)
    train_losses = np.load("train_losses.npy")
    return model, eval_data, train_losses

model, eval_data, train_losses = load_assets()
X_test = eval_data['X_test']
y_test = eval_data['y_test'].numpy()
y_preds = eval_data['test_preds']
y_probs = eval_data['test_probs'][:, 1] # Probabilities for "Right Hand" (class 1)

st.title("🧠 Scalable Cloud EEG Classification")

# Create Tabs
tab1, tab2 = st.tabs(["🚀 Live Simulation", "📊 Model Evaluation & Metrics"])

# --- TAB 1: LIVE SIMULATION ---
with tab1:
    st.markdown("Simulating Real-Time Cloud Inference for Motor Imagery Intent.")
    col1, col2 = st.columns([2, 1])
    
    sample_id = st.slider("Select Patient Test Sample", 0, len(X_test)-1, 0)
    run_btn = st.button("Start Inference")
    
    chart_placeholder = col1.empty()
    status_text = col2.empty()
    result_box = col2.empty()
    
    if run_btn:
        sample_signal = X_test[sample_id]
        true_label = "Left Hand" if y_test[sample_id] == 0 else "Right Hand"
        
        for i in range(10, len(sample_signal[0]), 50):
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(sample_signal[0, :i].numpy(), color='#1f77b4')
            ax.set_ylim([-100, 100])
            ax.set_title("Streaming Patient Data to Cloud")
            chart_placeholder.pyplot(fig)
            plt.close(fig)
            time.sleep(0.05)
            
        with torch.no_grad():
            output = model(sample_signal.unsqueeze(0))
            prediction = torch.argmax(output).item()
            
        pred_label = "Left Hand" if prediction == 0 else "Right Hand"
        status_text.write("✅ Analysis Complete")
        result_box.metric("Predicted Intent", pred_label, delta="Match" if pred_label == true_label else "Error")

# --- TAB 2: METRICS & EXPLANATIONS ---
with tab2:
    st.header("Model Evaluation Analytics")
    st.markdown("Evaluating machine learning models goes beyond simple accuracy. Here is how our architecture performed across advanced metrics.")
    
    mcol1, mcol2 = st.columns(2)
    
    # Confusion Matrix
    with mcol1:
        st.subheader("1. Confusion Matrix")
        
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Left', 'Right'], yticklabels=['Left', 'Right'])
        ax_cm.set_xlabel('Predicted Identity')
        ax_cm.set_ylabel('True Identity')
        st.pyplot(fig_cm)
        st.info("""
        **What this shows:** A matrix detailing where the model gets confused. 
        - **True Positives/Negatives (Diagonal):** Times the model correctly guessed Left or Right.
        - **False Positives/Negatives:** Times the model thought the patient imagined the Right hand, but they actually imagined the Left.
        """)

    # ROC Curve
    with mcol2:
        st.subheader("2. ROC Curve & AUC")
        
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
        st.info("""
        **What this shows:** The Receiver Operating Characteristic curve graphs the trade-off between sensitivity and specificity. 
        - **AUC (Area Under the Curve):** A score from 0 to 1. A score of 0.5 is no better than random guessing. The closer to 1.0, the better the model distinguishes between Left and Right hands.
        """)

    st.divider()

    # Classification Report
    st.subheader("3. Precision, Recall, and F1-Score")
    report = classification_report(y_test, y_preds, target_names=['Left Hand', 'Right Hand'], output_dict=True)
    
    rcol1, rcol2, rcol3 = st.columns(3)
    rcol1.metric("Precision (Right Hand)", f"{report['Right Hand']['precision']:.2f}")
    rcol2.metric("Recall (Right Hand)", f"{report['Right Hand']['recall']:.2f}")
    rcol3.metric("F1-Score (Weighted)", f"{report['weighted avg']['f1-score']:.2f}")
    
    st.markdown("""
    
    *   **Precision:** Out of all the times the Cloud Model declared "Right Hand", how many times was it actually correct? (Focuses on minimizing false alarms).
    *   **Recall (Sensitivity):** Out of all the actual times the patient thought about their Right Hand, how many did the model successfully catch? (Focuses on minimizing missed signals).
    *   **F1-Score:** The harmonic mean of Precision and Recall. It is the best metric for judging the overall health of a model when raw Accuracy is misleading.
    """)