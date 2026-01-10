import tqdm

from src.process.metrics import *
import torch


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, num_classes) -> dict[str, float]:
    model.eval()

    losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    f1s_weighted = []

    for images, labels in tqdm.tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        logits = model(images)
        loss = criterion(logits, labels)

        m = metrics(logits, labels, num_classes)

        losses.append(loss.detach())
        accuracies.append(m["accuracy"])
        precisions.append(m["precision"])
        recalls.append(m["recall"])
        f1s.append(m["f1"])
        f1s_weighted.append(m["f1_w"])

    losses = torch.stack(losses)
    accuracies = torch.stack(accuracies)
    precisions = torch.stack(precisions)
    recalls = torch.stack(recalls)
    f1s = torch.stack(f1s)
    f1s_weighted = torch.stack(f1s_weighted)

    return {
        "loss": losses.mean().item(),
        "accuracy": accuracies.mean().item(),
        "precision": precisions.mean().item(),
        "recall": recalls.mean().item(),
        "f1": f1s.mean().item(),
        "f1_w": f1s_weighted.mean().item()
    }
