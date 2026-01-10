import torch

@torch.no_grad()
def metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int, eps: float = 1e-8) -> dict:
    predictions = logits.argmax(dim=1)

    # accuracy
    accuracy = (predictions == labels).float().mean()

    # counts per class (GPU)
    pred_count = torch.bincount(predictions, minlength=num_classes).float()
    true_count = torch.bincount(labels, minlength=num_classes).float()

    correct_mask = (predictions == labels)
    tp = torch.bincount(labels[correct_mask], minlength=num_classes).float()

    fp = pred_count - tp
    fn = true_count - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    precision_macro = precision.mean()
    recall_macro = recall.mean()
    f1_macro = f1.mean()

    # weighted (ważone supportem = true_count)
    weight_sum = true_count.sum().clamp_min(eps)
    f1_weighted = (f1 * true_count).sum() / weight_sum

    return {
        "accuracy": accuracy,
        "precision": precision_macro,
        "recall": recall_macro,
        "f1": f1_macro,
        "f1_w": f1_weighted,
    }
