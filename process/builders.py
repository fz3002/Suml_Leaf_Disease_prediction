from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import yaml
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from model.model import SqueezeNet
from data_preparation.data_handler.datahandler import DataHandler


def build_optimizer(name: str, params, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer):
    sch_cfg = cfg.get("scheduler", {}) or {}
    if not sch_cfg.get("enable", False):
        return None

    sch_type = str(sch_cfg.get("type", "")).lower()

    if sch_type == "steplr":
        step_size = int(sch_cfg["step_size"])
        gamma = float(sch_cfg["gamma"])
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if sch_type == "multisteplr":
        milestones = sch_cfg["milestones"]
        gamma = float(sch_cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    if sch_type == "cosineannealinglr":
        t_max = int(sch_cfg.get("t_max", 10))
        eta_min = float(sch_cfg.get("eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    raise ValueError(f"Unknown scheduler type: {sch_cfg.get('type')}")


@dataclass
class TrainingParams:
    cfg: Dict[str, Any]
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    model: SqueezeNet
    device: torch.device
    criterion: nn.CrossEntropyLoss
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler
    run_name: str
    epochs: int


class PrepareTraining:
    """
    Ładuje config.yaml i buduje wszystkie obiekty potrzebne do treningu.
    """

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg: Dict[str, Any] = yaml.safe_load(f)

    def build(self) -> TrainingParams:
        cfg = self.cfg

        handler: DataHandler = DataHandler()

        train_loader: DataLoader = handler.get_dataloader('train')
        val_loader: DataLoader = handler.get_dataloader('val')
        test_loader: DataLoader = handler.get_dataloader('test')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model: SqueezeNet = SqueezeNet(
            num_classes=int(cfg["model"]["num_classes"]),
            dropout=float(cfg["model"]["dropout"]),
            init_weights=bool(cfg["model"]["init_weights"]),
        )

        model.to(device)

        # Criterion (klasyfikacja wieloklasowa)
        criterion: CrossEntropyLoss = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Optimizer
        optimizer = build_optimizer(
            name=str(cfg["train"]["optimizer"]),
            params=model.parameters(),
            lr=float(cfg["train"]["learning_rate"]),
            weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
        )

        # Scheduler
        scheduler = build_scheduler(cfg["train"], optimizer)

        run_name = str(cfg["train"].get("run_name", "run"))
        epochs = int(cfg["train"]["epochs"])

        return TrainingParams(
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model=model,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            run_name=run_name,
            epochs=epochs,
        )
