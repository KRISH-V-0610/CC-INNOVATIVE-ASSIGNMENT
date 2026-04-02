import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from model import EEG_CNN_RNN

def generate_visualizations():
    # 1. Load Data and Model
    X_test, y_test = torch.load("test_data.pth", weights_only=True)
    train_losses = np.load("train_losses.npy")
    
    model = EEG_CNN_RNN(num_channels=64, num_classes=2)
    model.load_state_dict(torch.load("eeg_model.pth", weights_only=True))
    model.eval()

    # Get Predictions
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)

    labels = ['Left Hand', 'Right Hand']
    
    # --- PLOT 1: Training Loss Curve ---
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, color='blue', linewidth=2)
    plt.title('Model Training Loss', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Cross Entropy)')
    plt.grid(True, linestyle='--', alpha=0.6)

    # --- PLOT 2: Confusion Matrix ---
    plt.subplot(1, 3, 2)
    cm = confusion_matrix(y_test.numpy(), predicted.numpy())
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Prediction Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # --- PLOT 3: Raw EEG Signal vs Prediction ---
    # Pick a random sample from the test set to visualize
    sample_idx = 0
    sample_signal = X_test[sample_idx, 0, :].numpy() # Just plotting the first electrode
    true_label = labels[y_test[sample_idx].item()]
    pred_label = labels[predicted[sample_idx].item()]

    plt.subplot(1, 3, 3)
    # Creating a dummy time axis for the plot
    time = np.linspace(-1, 4, len(sample_signal))
    plt.plot(time, sample_signal, color='purple', alpha=0.7)
    
    title_color = 'green' if true_label == pred_label else 'red'
    plt.title(f'Sample Prediction\nTrue: {true_label} | Pred: {pred_label}', 
              fontsize=14, color=title_color)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (µV)')
    
    plt.tight_layout()
    plt.savefig('assignment_results.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    generate_visualizations()