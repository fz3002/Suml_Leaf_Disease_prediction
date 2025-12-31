import json
import os
from pathlib import Path

from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from metrics import *
from src.data_preparation.data_handler.datahandler import DataHandler
from src.data_preparation.preprocessing.data_util import load_config_file
from src.model.model import SqueezeNet
from validate import validate_one_epoch


def train_one_epoch(model, loader, optimizer, criterion, device, num_classes) -> dict[str, float]:
    model.train()

    losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    f1s_weighted = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        m = metrics(logits.detach(), labels, num_classes)

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


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_metric: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "best_metric": float(best_metric),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(payload, path)


def save_metrics(path: Path, train_metrics: dict, val_metrics: dict, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "train": train_metrics,
        "val": val_metrics,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_weights(path: Path, model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def fit(model_section: dict, train_section: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(Path(output_dir) / "checkpoints", exist_ok=True)
    os.makedirs(Path(output_dir) / "metrics", exist_ok=True)
    os.makedirs(Path(output_dir) / "weights", exist_ok=True)

    num_classes = model_section["num_classes"]

    handler = DataHandler()
    train_loader = handler.get_dataloader('train')
    val_loader = handler.get_dataloader('val')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SqueezeNet(num_classes=model_section["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_section["learning_rate"],
        weight_decay=train_section["weight_decay"]
    )
    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=train_section["epochs"],
        eta_min=5e-7
    )

    print("=== Training config ===")
    print(f"device: {device}")
    print(f"epochs: {train_section['epochs']}")
    print(f"lr: {train_section['learning_rate']}")
    print(f"weight_decay: {train_section['weight_decay']}")
    print(f"scheduler: CosineAnnealingLR (eta_min={5e-7})")
    print("========================\n")

    val_loss: float = 10.0

    for epoch in range(1, train_section["epochs"] + 1):
        print(f"\nEpoch {epoch:02d}/{train_section["epochs"]} | lr={optimizer.param_groups[0]['lr']:.2e}")
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_classes=num_classes
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes
        )

        scheduler.step()
        print(f"train | loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} f1={train_metrics['f1']:.4f}")
        print(f"val   | loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f}")

        save_checkpoint(path=Path(output_dir) / "checkpoints" / f"epoch_{epoch}.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        best_metric=val_metrics["loss"])
        save_metrics(path=Path(output_dir) / "metrics" / f"epoch_{epoch}.json",
                     train_metrics=train_metrics,
                     val_metrics=val_metrics,
                     epoch=epoch)
        save_weights(path=Path(output_dir) / "weights" / f"epoch_{epoch}.pt",
                     model=model)

        if val_loss > val_metrics["loss"]:
            val_loss = val_metrics["loss"]
            save_weights(path=Path(output_dir) / "weights" / f"best.pt",
                         model=model)


if __name__ == '__main__':
    config_content = load_config_file()
    project_path = config_content["path"]["project_path"]
    model_weights = config_content["path"]["model_weights"]["root"]
    run_name = config_content["train"]["run_name"]
    output_dir = os.path.join(project_path, model_weights, run_name)
    fit(model_section=config_content["model"],
        train_section=config_content["train"],
        output_dir=str(output_dir))

