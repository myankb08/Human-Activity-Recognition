import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from sklearn.metrics import classification_report
from src import data_loader, models

def train_model(model, train_loader, criterion, optimizer, device, n_epochs=20):
    model.to(device)
    model.train()
    print(f"--- Training {model.__class__.__name__} ---")
    for epoch in range(n_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return classification_report(all_labels, all_preds, output_dict=True)

def run_experiment():
    print("--- Starting PyTorch HAR Experiment ---")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(BASE_DIR, 'data')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, test_loader = data_loader.get_dataloaders(dataset_path)
    all_reports = {}
    
    model_defs = {'MLP': models.MLP(), 'CNN': models.CNN(), 'LSTM': models.LSTM_Net(), 'CNN-LSTM': models.CNN_LSTM()}

    for name, model in model_defs.items():
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_model(model, train_loader, criterion, optimizer, device)
        report = evaluate_model(model, test_loader, device)
        all_reports[name] = report
    
    results_data = {model_name: {'Accuracy': r['accuracy'], 'Precision': r['macro avg']['precision'],
                                 'Recall': r['macro avg']['recall'], 'F1-Score': r['macro avg']['f1-score']}
                    for model_name, r in all_reports.items()}
    
    results_df = pd.DataFrame(results_data).T
    print("\n--- Comparative Performance Metrics ---")
    print(results_df)

if __name__ == '__main__':
    run_experiment()