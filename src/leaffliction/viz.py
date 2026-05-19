"""Matplotlib chart helpers used across all 5 entrypoints."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PALETTE = sns.color_palette("Set2")


def pie_and_bar(class_counts: dict[str, int], title: str, save: Path | None = None) -> None:
    """Render a side-by-side pie+bar of class distribution (Part 1)."""
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(classes))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].pie(counts, labels=classes, autopct="%1.1f%%", colors=colors, startangle=90)
    axes[0].set_title(f"{title} class distribution")

    bars = axes[1].bar(classes, counts, color=colors)
    axes[1].set_xticks(range(len(classes)))
    axes[1].set_xticklabels(classes, rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.3)
    # Annotate each bar with its exact count.
    for bar, n in zip(bars, counts, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(n),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def grid(images: list[tuple[str, np.ndarray]], save: Path | None = None) -> None:
    """Render N labelled images in a single row (used by Augmentation/Transformation)."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, (label, img) in zip(axes, images, strict=True):
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def learning_curves(history: dict[str, list[float]], save: Path) -> None:
    """Two-axis loss/accuracy plot for train.py."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"], label="val")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    plt.tight_layout()
    Path(save).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save, dpi=120, bbox_inches="tight")
    plt.close(fig)


def confusion_matrix_plot(cm: np.ndarray, classes: list[str], save: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save, dpi=120, bbox_inches="tight")
    plt.close(fig)
