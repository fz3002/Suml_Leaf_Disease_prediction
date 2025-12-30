from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from builders import PrepareTraining, TrainingParams


@dataclass
class EpochResult:
    loss: float
    acc: float

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, grad_clip: float | None = None) -> EpochResult:
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        if i == 0:
            w_before = model.final_conv.weight.detach().clone()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            if i == 0:
                w_after = model.final_conv.weight.detach()
                print("Δw (final_conv) mean:", (w_after - w_before).abs().mean().item())
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            if i == 0:
                w_after = model.final_conv.weight.detach()
                print("Δw (final_conv) mean:", (w_after - w_before).abs().mean().item())

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits.detach(), y) * bs
        n += bs

    return EpochResult(loss=total_loss / n, acc=total_acc / n)

@torch.no_grad()
def evaluate(model, loader, criterion, device) -> EpochResult:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    c = Counter()

    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()
        logits = model(x)
        loss = criterion(logits, y)

        preds = logits.argmax(dim=1).cpu().tolist()
        c.update(preds)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits, y) * bs
        n += bs

    print(c)
    return EpochResult(loss=total_loss / n, acc=total_acc / n)

def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_metric: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
    }, path)

def fit(bundle: TrainingParams, use_amp: bool = True, grad_clip: float | None = 1.0):
    project_path = Path(bundle.cfg["path"]["project_path"])
    weight_path = Path(bundle.cfg["path"]["model_weights"]["root"])
    run_path: Path = project_path / weight_path / bundle.run_name
    run_path.mkdir(parents=True, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler() if (use_amp and bundle.device.type == "cuda") else None

    best_val_acc = -1.0

    for epoch in range(1, bundle.epochs + 1):
        train_res = train_one_epoch(
            bundle.model, bundle.train_loader, bundle.optimizer,
            bundle.criterion, bundle.device, scaler=scaler, grad_clip=grad_clip
        )
        val_res = evaluate(bundle.model, bundle.val_loader, bundle.criterion, bundle.device)

        # scheduler
        if bundle.scheduler is not None:
            if isinstance(bundle.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                bundle.scheduler.step(val_res.loss)
            else:
                bundle.scheduler.step()

        lr = bundle.optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch:02d}/{bundle.epochs} | lr={lr:.2e} | "
              f"train_loss={train_res.loss:.4f} acc={train_res.acc:.4f} | "
              f"val_loss={val_res.loss:.4f} acc={val_res.acc:.4f}")

        # save last
        save_checkpoint(run_path / f"epoch_{epoch}.pt", bundle.model, bundle.optimizer, bundle.scheduler, epoch, best_val_acc)

        # save best
        if val_res.acc > best_val_acc:
            best_val_acc = val_res.acc
            save_checkpoint(run_path / "best.pt", bundle.model, bundle.optimizer, bundle.scheduler, epoch, best_val_acc)

    # final test
    test_res = evaluate(bundle.model, bundle.test_loader, bundle.criterion, bundle.device)
    print(f"TEST | loss={test_res.loss:.4f} acc={test_res.acc:.4f}")

def inspect_dataloader(loader, name="loader", max_batches=None):
    print(f"\n=== Inspecting {name} ===")
    print(f"Total batches: {len(loader)}")

    for i, (x, y) in enumerate(loader):
        print(f"\nBatch {i}:")
        print(f"  images shape: {x.shape}")
        print(f"  labels shape: {y.shape}")
        print(f"  labels dtype: {y.dtype}")
        print(f"  unique labels in batch: {torch.unique(y).tolist()}")
        print(f"  num unique labels: {torch.unique(y).numel()}")

        if max_batches is not None and i + 1 >= max_batches:
            break


def inspect_labels_from_loader(loader, name="loader", max_batches=None):
    print(f"\n=== Inspect labels from {name} ===")
    all_labels = []

    for i, (_, y) in enumerate(loader):
        all_labels.extend(y.tolist())

        if max_batches is not None and i + 1 >= max_batches:
            break

    c = Counter(all_labels)
    print("label distribution:", c)
    print("num classes seen:", len(c))
    print("min label:", min(c), "max label:", max(c))


if __name__ == '__main__':
    prepare = PrepareTraining(config_path="/home/michal/PycharmProjects/Suml_Leaf_Disease_prediction/config.yaml")
    params = prepare.build()

    fit(params, use_amp=False)

