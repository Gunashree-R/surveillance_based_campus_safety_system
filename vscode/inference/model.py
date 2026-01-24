import torch
import torch.nn as nn

class VideoClassifier(nn.Module):
    def __init__(self, input_size=512, hidden_size=128, num_classes=9):
        super(VideoClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out
