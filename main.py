from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from model import EEG_CNN_RNN
import uvicorn

app = FastAPI(title="Cloud EEG Inference Engine")

# This allows your Vercel frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, you will change this to your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the AI and Data into memory on server startup
print("Loading Model to Cloud API...")
model = EEG_CNN_RNN(num_channels=64, num_classes=2)
model.load_state_dict(torch.load("eeg_model.pth", weights_only=True))
model.eval()

# Load our evaluation data to simulate the streaming source
eval_data = torch.load("eval_data.pth", weights_only=False)
X_test = eval_data['X_test']
y_test = eval_data['y_test'].numpy()

@app.get("/")
def health_check():
    return {"status": "Cloud AI Engine is Online"}

@app.get("/api/predict/{sample_id}")
def predict_eeg(sample_id: int):
    # 1. Get the requested patient data
    if sample_id < 0 or sample_id >= len(X_test):
        return {"error": "Patient sample not found"}
        
    sample_signal = X_test[sample_id]
    true_label = "Left Hand" if int(y_test[sample_id]) == 0 else "Right Hand"
    
    # 2. Run Cloud Inference
    with torch.no_grad():
        output = model(sample_signal.unsqueeze(0))
        probabilities = torch.softmax(output, dim=1)[0].tolist()
        prediction_idx = int(torch.argmax(output).item())
        
    pred_label = "Left Hand" if prediction_idx == 0 else "Right Hand"
    confidence = probabilities[prediction_idx] * 100

    # 3. Return the prediction AND a slice of the raw signal for the frontend to draw
    return {
        "sample_id": sample_id,
        "true_intent": true_label,
        "predicted_intent": pred_label,
        "confidence": f"{confidence:.2f}%",
        "signal_data": sample_signal[0, :].tolist() # Channel 0 data for the chart
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)