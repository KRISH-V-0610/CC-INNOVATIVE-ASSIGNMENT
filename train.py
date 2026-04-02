import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import load_and_prep_data
from model import EEG_CNN_RNN

def train_model():
    X_train, X_test, y_train, y_test = load_and_prep_data()
    
    model = EEG_CNN_RNN(num_channels=64, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 60 # Increased epochs for larger dataset
    train_losses = []

    print(f"\nStarting Training on {len(X_train)} samples...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Run predictions on test set to save evaluation metrics
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_probs = torch.softmax(test_outputs, dim=1).numpy()
        test_preds = torch.argmax(test_outputs, dim=1).numpy()

    # Save Everything
    torch.save(model.state_dict(), "eeg_model.pth")
    np.save("train_losses.npy", train_losses)
    torch.save({
        'X_test': X_test,
        'y_test': y_test,
        'test_probs': test_probs,
        'test_preds': test_preds
    }, "eval_data.pth")
    
    print("Training Complete. Model and evaluation data saved.")

if __name__ == "__main__":
    train_model()