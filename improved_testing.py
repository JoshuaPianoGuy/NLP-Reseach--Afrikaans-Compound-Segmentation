import torch
import numpy as np
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

HUGGING_FACE_TOKEN = "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"

class CompoundSegmentationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.data = self.load_data(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_data(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    original, segmented = line.strip().split('\t')
                    data.append((original, segmented))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'original': self.data[idx][0],
            'segmented': self.data[idx][1]
        }

def evaluate_model(model, dataset, tokenizer, device):
    model.eval()
    all_predictions = []
    all_originals = []
    all_true_segmentations = []
    
    for item in dataset:
        original = item['original']
        true_segmentation = item['segmented']
        
        # Generate all possible split points for this word
        candidates = [f"{original} @ {original[:i]}@{original[i:]}" 
                     for i in range(1, len(original))]
        
        if not candidates:  # Handle edge case of very short words
            all_predictions.append(original)
            all_originals.append(original)
            all_true_segmentations.append(true_segmentation)
            continue
            
        # Tokenize all candidates
        inputs = tokenizer(candidates, 
                         return_tensors="pt", 
                         padding=True, 
                         truncation=True,
                         max_length=128).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs.logits.argmax(1).cpu().tolist()
        
        # Construct predicted segmentation
        predicted = list(original)
        for i, pred in enumerate(preds, 1):
            if pred == 1:
                predicted.insert(i, '@')
        predicted = ''.join(predicted)
        
        all_predictions.append(predicted)
        all_originals.append(original)
        all_true_segmentations.append(true_segmentation)
    
    # Calculate metrics
    correct = sum(1 for pred, true in zip(all_predictions, all_true_segmentations) if pred == true)
    accuracy = correct / len(all_predictions)
    
    # Calculate boundary-based metrics
    TP = FP = FN = 0
    for pred, true in zip(all_predictions, all_true_segmentations):
        pred_boundaries = set([i for i, char in enumerate(pred) if char == '@'])
        true_boundaries = set([i for i, char in enumerate(true) if char == '@'])
        
        TP += len(pred_boundaries & true_boundaries)
        FP += len(pred_boundaries - true_boundaries)
        FN += len(true_boundaries - pred_boundaries)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': list(zip(all_originals, all_predictions, all_true_segmentations))
    }

def test_fine_tuned_model(lang):
    # Load the fine-tuned model and tokenizer
    model_path = f"/home/jdiegaardt/lustre/NLP_afri/out/trained-model-{lang}-def"
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=HUGGING_FACE_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, token=HUGGING_FACE_TOKEN)
    
    # Load the test dataset
    test_dataset = CompoundSegmentationDataset(f"./Dataset/{lang}_test.txt", tokenizer)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Evaluate the model
    results = evaluate_model(model, test_dataset, tokenizer, device)
    
    # Print metrics
    print(f"Test Accuracy: {results['accuracy'] * 100:.2f}%")
    print(f"Precision: {results['precision'] * 100:.2f}%")
    print(f"Recall: {results['recall'] * 100:.2f}%")
    print(f"F1 Score: {results['f1'] * 100:.2f}%")
    
    # Save metrics to JSON
    metrics = {
        "accuracy": results['accuracy'],
        "precision": results['precision'],
        "recall": results['recall'],
        "f1_score": results['f1']
    }
    
    with open(f"{model_path}/test_results_improved.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Save predictions to CSV
    predictions_data = []
    for original, predicted, true_segmentation in results['predictions']:
        pred_boundaries = [i for i, char in enumerate(predicted) if char == '@']
        true_boundaries = [i for i, char in enumerate(true_segmentation) if char == '@']
        
        predictions_data.append({
            "original_word": original,
            "predicted_segmentation": predicted,
            "true_segmentation": true_segmentation,
            "predicted_boundaries": pred_boundaries,
            "true_boundaries": true_boundaries
        })
    
    df = pd.DataFrame(predictions_data)
    df.to_csv(f"{model_path}/predictions_improved.csv", index=False)

if __name__ == "__main__":
    test_fine_tuned_model("afrikaans")