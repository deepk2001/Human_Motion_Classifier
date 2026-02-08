from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset

#TODO:Inherit from your desired classifier to create a custom classifier.
class TraditionalClassifier():
    def __init__(self,k=5):
        # TODO: Pass the parameters you require to your selected classifier
        super().__init__()
        self.k = k
        self.classifier = KNeighborsClassifier(n_neighbors=k)
    
    def fit(self, X, y):
        self.classifier.fit(X, y)
    
    def predict(self, X):
        return self.classifier.predict(X)

    def evaluate(self,train_feats, train_labels, test_feats, test_labels):
        """
        TODO:
        1. Predict labels for the training set.
        2. Predict labels for the testing set.
        3. Compute training accuracy using accuracy_score.
        4. Compute testing accuracy using accuracy_score.
        5. Print the training and testing accuracy percentages.
        """
        pred_train = self.classifier.predict(train_feats)
        pred_test = self.classifier.predict(test_feats)

        # Accuracy
        train_acc = accuracy_score(train_labels, pred_train) * 100
        test_acc = accuracy_score(test_labels, pred_test) * 100

        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Testing Accuracy:  {test_acc:.2f}%")

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, activation=nn.ReLU, num_layers=2):
        super(MLP, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(CNN, self).__init__()
        
        # We treat the input_dim as the 'sequence length' and 
        # assume 1 input channel (treating the feature vector as a 1D signal)
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Reduces any input length to 1
        )
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x shape: (Batch, Features)
        # CNN1D needs: (Batch, Channels, Features)
        x = x.unsqueeze(1) 
        
        x = self.conv_block(x) # Shape: (Batch, hidden_dim*2, 1)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x