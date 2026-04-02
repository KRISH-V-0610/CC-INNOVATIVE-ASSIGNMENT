import torch
import torch.nn as nn

class EEG_CNN_RNN(nn.Module):
    def __init__(self, num_channels=64, num_classes=2):
        super(EEG_CNN_RNN, self).__init__()
        self.cnn = nn.Conv1d(in_channels=num_channels, out_channels=16, kernel_size=32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=4)
        
        # New: Dropout layer drops 50% of connections randomly during training
        # This forces the model to generalize and not memorize specific brainwaves.
        self.dropout = nn.Dropout(0.5) 
        
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        out, (hn, cn) = self.lstm(x)
        last_time_step = out[:, -1, :]
        predictions = self.fc(last_time_step)
        return predictions