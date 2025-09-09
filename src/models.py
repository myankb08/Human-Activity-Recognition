import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_features=9, n_timesteps=128, n_outputs=6):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_features * n_timesteps, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, n_outputs)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.fc3(x)

class CNN(nn.Module):
    def __init__(self, n_features=9, n_outputs=6):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 62, 100) 
        self.fc2 = nn.Linear(100, n_outputs)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class LSTM_Net(nn.Module):
    def __init__(self, n_features=9, n_outputs=6, n_hidden=100, n_layers=1):
        super(LSTM_Net, self).__init__()
        self.lstm = nn.LSTM(n_features, n_hidden, n_layers, batch_first=True, dropout=0.5)
        self.fc1 = nn.Linear(n_hidden, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, n_outputs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class CNN_LSTM(nn.Module):
    def __init__(self, n_features=9, n_outputs=6, n_hidden=100):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=n_hidden, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(n_hidden, 100)
        self.fc2 = nn.Linear(100, n_outputs)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        _, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])
        x = self.relu(self.fc1(x))
        return self.fc2(x)