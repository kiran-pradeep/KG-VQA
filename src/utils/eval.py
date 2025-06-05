import torch
import torch.nn as nn
import numpy as np
import json
import argparse
from torch.utils.data import Dataset, DataLoader
import os

# Load JSONL (actually JSON in your case)
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# Dataset class (same as training)
class DecompositionDataset(Dataset):
    def __init__(self, datapoints):
        self.X = np.array([[dp['average_question_bleu'], 
                            dp['average_context_bleu'],
                            dp['average_question_distance'], 
                            dp['average_context_distance']] for dp in datapoints], dtype=np.float32)
        self.metadata = [dp.get("id", i) for i, dp in enumerate(datapoints)]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), self.metadata[idx]

# Model definition (same as training)
class ForcePredictor(nn.Module):
    def __init__(self):
        super(ForcePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

def evaluate(model_path, dev_path, batch_size=32):
    model = ForcePredictor()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    data = load_json(dev_path)
    dataset = DecompositionDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_preds = []
    with torch.no_grad():
        for X, _ in loader:
            preds = model(X).squeeze()
            all_preds.extend(preds.tolist())

    return all_preds

def summarize(scores, label):
    mean_score = np.mean(scores)
    print(f"🔍 {label} Mean Predicted Score: {mean_score:.4f}")
    return mean_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="force_predictor.pt")
    parser.add_argument("--clean_path", type=str, default="results/dev/w_bleu=1.0_w_sem=0.4/-scores.json")
    parser.add_argument("--corrupt_path", type=str, default="results/dev_corrupted/w_bleu=1.0_w_sem=0.4/-scores.json")
    args = parser.parse_args()

    clean_scores = evaluate(args.model_path, args.clean_path)
    corrupt_scores = evaluate(args.model_path, args.corrupt_path)

    summarize(clean_scores, "Clean Dev Set")
    summarize(corrupt_scores, "Corrupted Dev Set")
