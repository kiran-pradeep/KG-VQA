import copy
import os
import json
import argparse
import random
import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from utils.qd_metrics import calculate_scores, preprocess_instance, init_worker, perturb_data
import multiprocessing as mp
from tqdm import tqdm

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

FEATURES = [
    "average_question_similarity",
    "average_context_similarity",
    "average_question_precision",
    "average_context_precision",
    "recall"
]


class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def load_data(path):
    with open(path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return [preprocess_instance(ex) for ex in data]


def generate_labeled_data(instances, mode):
    print(f"🔍 Calculating features for {mode} data")
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=os.cpu_count() - 4, initializer=init_worker) as pool:
        feature_data = list(tqdm(
            pool.imap_unordered(calculate_scores, instances),
            total=len(instances),
        ))
    for item in feature_data:
        item["label"] = 1
    return feature_data


def generate_noisy_data(instances):
    random.shuffle(instances)
    n = len(instances)
    n_40 = int(0.4 * n)
    n_20 = int(0.2 * n)

    split1 = instances[:n_40]
    split2 = instances[n_40:n_40*2]
    split3 = instances[n_40*2:]

    split1 = perturb_data(split1, add_irrelevant=0.5)
    split2 = perturb_data(split2, remove_relevant=0.5)
    split3 = perturb_data(split3, remove_relevant=0.3, add_irrelevant=0.3)

    noisy_instances = split1 + split2 + split3

    print(f"🔁 Generating noisy data...")
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=os.cpu_count() - 4, initializer=init_worker) as pool:
        noisy_data = list(tqdm(
            pool.imap_unordered(calculate_scores, noisy_instances),
            total=len(noisy_instances)
        ))

    for item in noisy_data:
        item["label"] = 0

    return noisy_data


def extract_features_labels(data):
    X = [[d[feat] for feat in FEATURES] for d in data]
    y = [d["label"] for d in data]
    return np.array(X), np.array(y)


def main(args):
    print("📂 Loading datasets...")
    train_data = load_data(args.train_path)
    dev_data = load_data(args.dev_path)

    clean_train = generate_labeled_data(train_data, "train")
    noisy_train = generate_noisy_data(copy.deepcopy(train_data))

    combined_train = clean_train + noisy_train
    random.shuffle(combined_train)

    X_train, y_train = extract_features_labels(combined_train)

    clean_dev = generate_labeled_data(dev_data, "dev")
    noisy_dev = generate_noisy_data(copy.deepcopy(dev_data))
    combined_dev = clean_dev + noisy_dev
    random.shuffle(combined_dev)
    X_dev, y_dev = extract_features_labels(combined_dev)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.transform(X_dev)

    print("🚀 Training Feedforward Neural Network...")

    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_dev_tensor = torch.tensor(X_dev, dtype=torch.float32)
    y_dev_tensor = torch.tensor(y_dev, dtype=torch.float32).unsqueeze(1)

    # Dataloaders
    batch_size = 32
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

    # Model
    model = NeuralNet(input_size=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 20
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_dev_tensor)
            val_preds = (val_outputs > 0.5).float()
            val_loss = criterion(val_outputs, y_dev_tensor).item()

            val_accuracy = accuracy_score(y_dev, val_preds.numpy())
            val_precision = precision_score(y_dev, val_preds.numpy())
            val_recall = recall_score(y_dev, val_preds.numpy())

        print(f"Epoch {epoch+1:02d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, "
              f"Accuracy = {val_accuracy:.4f}, Precision = {val_precision:.4f}, Recall = {val_recall:.4f}")

    # Final Evaluation
    print("\n📊 Final Evaluation on Dev Set:")
    cm = confusion_matrix(y_dev, val_preds.numpy())
    print("Confusion Matrix:\n", cm)
    print(classification_report(y_dev, val_preds.numpy(), digits=4))

    # Save model and scaler
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/neural_model.pt")
    torch.save(scaler, "outputs/feature_scaler.pt")
    print("✅ Model saved to outputs/neural_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/MuSiQue/updated_train.jsonl", help="Path to the train JSON file")
    parser.add_argument("--dev_path", type=str, default="data/MuSiQue/updated_dev.jsonl", help="Path to the dev JSON file")

    args = parser.parse_args()
    main(args)
