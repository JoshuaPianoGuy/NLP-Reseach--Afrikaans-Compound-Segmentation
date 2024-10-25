import json
from sklearn.metrics import precision_score, recall_score, f1_score

# Function to convert boundaries (hyphens or asterisks) into binary labels (1 for boundary, 0 otherwise)
def convert_to_binary_labels(segmentation):
    """
    Convert segmentation string to binary labels.
    Treats *, ., and - as boundaries.
    """
    # Normalize: Remove unwanted characters and treat them as boundaries
    # Here, we replace boundaries with a space for easy processing
    normalized_segmentation = segmentation.replace('*', '').replace('.', '').replace('-', '')

    # Split into characters and convert to binary: 1 for boundary, 0 for non-boundary
    binary_labels = []
    for char in normalized_segmentation:
        if char.strip():  # Ignore spaces
            binary_labels.append(1)  # Boundary found
        else:
            binary_labels.append(0)  # No boundary

    return binary_labels

# Function to evaluate model predictions
def evaluate(true_segmentations, predicted_segmentations):
    """Evaluate model predictions by comparing true and predicted segmentations."""
    
    true_labels = []
    predicted_labels = []
    
    for true_seg, pred_seg in zip(true_segmentations, predicted_segmentations):
        # Use the updated convert_to_binary_labels function
        true_binary = convert_to_binary_labels(true_seg)
        predicted_binary = convert_to_binary_labels(pred_seg)
        
        # Debugging information
        print(f"True Segmentation: {true_seg} | Length: {len(true_binary)}")
        print(f"Predicted Segmentation: {pred_seg} | Length: {len(predicted_binary)}")
        
        if len(true_binary) != len(predicted_binary):
            raise ValueError(f"Error: Mismatched lengths after processing. True: {len(true_binary)}, Predicted: {len(predicted_binary)}")
        
        true_labels.extend(true_binary)
        predicted_labels.extend(predicted_binary)
    
    # Calculate precision, recall, and F1 score based on boundary predictions
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics


# Load your true and predicted data from files (assuming you have preprocessed data ready)
with open("Data/val_dict.txt", "r") as f:
    true_segmentations = [line.strip() for line in f.readlines()]

with open("Data/patgen_results.txt", "r") as f:
    predicted_segmentations = [line.strip() for line in f.readlines()]

# Perform evaluation
try:
    metrics = evaluate(true_segmentations, predicted_segmentations)
    
    # Print out the results
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Write metrics to a JSON file
    with open("Data/eval_1.json", "w") as json_file:
        json.dump(metrics, json_file, indent=4)
        
    print("Metrics saved to Data/eval_1.json")

except ValueError as e:
    print(e)
