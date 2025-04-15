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

    probs = [trainer.get_probabilities(test_loader) for trainer in trainers]
    ensemble_probs = np.mean(probs, axis=0)
    ensemble_preds = [1 if p >= 0.5 else 0 for p in ensemble_probs]

    true_labels = []
    for _, labels, _ in test_loader:
        true_labels.extend(labels.numpy())

    misclassified = []
    for i, (true_label, pred, (_, _, path)) in enumerate(zip(true_labels, ensemble_preds, test_loader.dataset)):
        if true_label != pred:
            misclassified.append({
                'path': path,
                'true_label': true_label,
                'pred_label': pred
            })
    for item in misclassified:
        print(f"Path: {item['path']}, True label: {item['true_label']} (0=Artifact , 1=noArtifact), Predicted: {item['pred_label']}")
    
    return f1_score(true_labels, ensemble_preds, average='micro')
