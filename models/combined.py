import torch.nn as nn
from cnn import CNNFeatureExtractor
from lstm import LSTMClassifier

class CombinedModel(nn.Module):
    def __init__(self, cnn_input_size=(64, 64), lstm_input_dim=4096, lstm_hidden_dim=256, lstm_num_layers=2, num_classes=5):
        super(CombinedModel, self).__init__()
        self.cnn = CNNFeatureExtractor()
        self.lstm = LSTMClassifier(
            input_dim=lstm_input_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            num_classes=num_classes,
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, sequence_length, channels, height, width = x.shape
        cnn_features = []

        # Extract features from each frame using CNN
        for t in range(sequence_length):
            frame_features = self.cnn(x[:, t, :, :, :])  # CNN outputs 4096-dimensional vector
            cnn_features.append(frame_features)

        # Combine features into a sequence
        cnn_features = torch.stack(cnn_features, dim=1)  # Shape: (batch_size, sequence_length, 4096)

        # Pass the sequence to LSTM
        out = self.lstm(cnn_features)
        return out
