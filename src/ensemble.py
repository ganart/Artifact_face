import numpy as np
from sklearn.metrics import f1_score

def compute_ensemble(trainers, test_loader):
    """
    Compute ensemble predictions by averaging probabilities from multiple models.
    Args:
        trainers (list): List of Trainer objects for each model.
        test_loader (DataLoader): DataLoader for test data.
    Returns:
        float: F1 score of the ensemble predictions.
    """
    # Get probabilities from each model
    probs = [trainer.get_probabilities(test_loader) for trainer in trainers]
    ensemble_probs = np.mean(probs, axis=0)
    ensemble_preds = [1 if p >= 0.5 else 0 for p in ensemble_probs]
    
    # Get true labels from the test loader
    true_labels = []
    for batch in test_loader:
        # Assuming batch structure is (images, labels) or (images, labels, ...)
        labels = batch[1]
        true_labels.extend(labels.cpu().numpy())
    
    # Find misclassified examples
    misclassified = []
    
    # Get paths from dataset - assuming dataset has a method to access image paths
    paths = [test_loader.dataset[i][2] for i in range(len(test_loader.dataset))]
    
    for i, (true_label, pred, path) in enumerate(zip(true_labels, ensemble_preds, paths)):
        if true_label != pred:
            misclassified.append({
                'path': path,
                'true_label': true_label,
                'pred_label': pred
            })
    
    # Print misclassified examples
    for item in misclassified:
        print(f"Path: {item['path']}, True label: {item['true_label']} (0=Artifact, 1=noArtifact), Predicted: {item['pred_label']}")
    
    return f1_score(true_labels, ensemble_preds, average='micro'), misclassified
