import json
import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_metrics(path: str) -> dict[str, list]:
    data: dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "train_f1_w": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_f1_w": []
    }

    files = sorted(
        [fn for fn in os.listdir(path) if fn.lower().endswith(".json")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    train_keys = ["loss", "accuracy", "precision", "recall", "f1", "f1_w"]
    val_keys = ["loss", "accuracy", "precision", "recall", "f1", "f1_w"]

    for file in files:
        with open(os.path.join(path, file), "r", encoding="utf-8") as f:
            content = json.load(f)

        data["epoch"].append(content.get("epoch"))

        train = content.get("train", {}) or {}
        val = content.get("val", {}) or {}

        for k in train_keys:
            data[f"train_{k}"].append(train.get(k))
        for k in val_keys:
            data[f"val_{k}"].append(val.get(k))

    return data

def plot(title: str, name: str, values: dict, palette: dict) -> None:
    df = pd.DataFrame({
        "epoch": values["epoch"],
        "train": values["train"],
        "val": values["val"]
    })

    df_long = df.melt(
        id_vars="epoch",
        var_name="type",
        value_name="value"
    )

    sns.set_theme(style="darkgrid", context="paper")

    plt.figure(figsize=(6, 5))
    sns.lineplot(
        data=df_long,
        x="epoch",
        y="value",
        hue="type",
        linewidth=2,
        palette=palette,
        errorbar=None
    )

    plt.xlabel("Epoch")
    plt.ylabel(name)
    plt.title(title)

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    metrics = load_metrics(r"C:\Users\Michał\PycharmProjects\SUML\data\4_weights\final\metrics")
    print(metrics.keys())

    m = {
        "epoch": metrics["epoch"],
        "train": metrics["train_accuracy"],
        "val": metrics["val_accuracy"]
    }

    palette = {
        "train": "#264653",  # ciemny granat
        "val": "#e76f51",  # ciepły czerwono-pomarańczowy
    }

    plot(title="Train vs. Val Loss",
         name="Loss",
         values=m,
         palette=palette)
