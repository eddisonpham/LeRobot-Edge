"""Teacher → student distillation for LeRobot policies.

Trains a smaller student policy that imitates a larger teacher,
evaluated with the same benchmark harness from benchmark.py.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot_edge.base import DeploymentBackend, NativePyTorchBackend
from lerobot_edge.configs import EdgeBaseConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------


class DistillationLoss(nn.Module):
    """Combined MSE + KL divergence loss for knowledge distillation.

    The loss blends:
    - MSE regression against teacher outputs (action chunks)
    - KL divergence on action distributions (if teacher provides logits)

    Args:
        temperature: Softmax temperature for KL divergence.
        alpha: Weight for KL vs MSE loss (0 = pure MSE, 1 = pure KL).
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_actions: torch.Tensor,
        teacher_actions: torch.Tensor,
        student_logits: torch.Tensor | None = None,
        teacher_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute distillation loss.

        Args:
            student_actions: Student model action predictions.
            teacher_actions: Teacher model action predictions (targets).
            student_logits: Optional student logits for KL computation.
            teacher_logits: Optional teacher logits for KL computation.

        Returns:
            Combined distillation loss.
        """
        # MSE loss on action predictions
        mse_loss = F.mse_loss(student_actions, teacher_actions)

        if student_logits is not None and teacher_logits is not None:
            # KL divergence loss
            student_log_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
            kl_loss = F.kl_div(
                student_log_soft,
                teacher_soft,
                reduction="batchmean",
            ) * (self.temperature ** 2)

            # Combined loss
            return (1 - self.alpha) * mse_loss + self.alpha * kl_loss
        else:
            return mse_loss


# ---------------------------------------------------------------------------
# Distillation backend
# ---------------------------------------------------------------------------


class DistilledBackend(NativePyTorchBackend):
    """Deployment backend for distilled student models."""

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(student_model, device)
        self._teacher = teacher_model
        if self._teacher is not None:
            self._teacher.eval()
            for param in self._teacher.parameters():
                param.requires_grad = False

    @property
    def teacher(self) -> nn.Module | None:
        return self._teacher


# ---------------------------------------------------------------------------
# Distillation training loop
# ---------------------------------------------------------------------------


def distill(
    teacher: nn.Module,
    student: nn.Module,
    config: EdgeBaseConfig,
    train_dataloader: Any,
    val_dataloader: Any | None = None,
    *,
    num_epochs: int | None = None,
    learning_rate: float | None = None,
    temperature: float | None = None,
    alpha: float | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run knowledge distillation training.

    Args:
        teacher: Teacher model (frozen during training).
        student: Student model to train.
        config: Edge configuration.
        train_dataloader: Training data loader.
        val_dataloader: Optional validation data loader.
        num_epochs: Number of training epochs (overrides config).
        learning_rate: Learning rate (overrides config).
        temperature: Distillation temperature (overrides config).
        alpha: KL vs MSE weight (overrides config).
        device: Training device (overrides config).

    Returns:
        Dict with training metrics and the best student checkpoint.
    """
    # Resolve parameters
    epochs = num_epochs or config.distill_epochs
    lr = learning_rate or config.distill_lr
    temp = temperature or config.distill_temperature
    a = alpha or config.distill_alpha
    dev = device or torch.device(config.device or "cpu")

    logger.info(
        "Starting distillation: teacher=%s, student=%s, epochs=%d, lr=%.2e, temp=%.1f, alpha=%.2f",
        type(teacher).__name__,
        type(student).__name__,
        epochs,
        lr,
        temp,
        a,
    )

    # Move models to device
    teacher = teacher.to(dev).eval()
    student = student.to(dev).train()

    # Freeze teacher
    for param in teacher.parameters():
        param.requires_grad = False

    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = DistillationLoss(temperature=temp, alpha=a)

    # Training loop
    best_val_loss = float("inf")
    best_student_state = None
    metrics: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_dataloader:
            # Move batch to device
            batch = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Teacher forward pass (no gradients)
            with torch.no_grad():
                teacher_actions = teacher.select_action(batch)

            # Student forward pass
            student_actions = student.select_action(batch)

            # Compute loss
            loss = criterion(student_actions, teacher_actions)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()

        avg_train_loss = epoch_loss / max(num_batches, 1)
        metrics["train_loss"].append(avg_train_loss)
        metrics["learning_rate"].append(scheduler.get_last_lr()[0])

        # Validation
        val_loss = 0.0
        if val_dataloader is not None:
            val_loss = _validate(teacher, student, val_dataloader, criterion, dev)
            metrics["val_loss"].append(val_loss)

        epoch_time = time.time() - epoch_start

        logger.info(
            "Epoch %d/%d: train_loss=%.4f, val_loss=%.4f, time=%.1fs",
            epoch + 1,
            epochs,
            avg_train_loss,
            val_loss,
            epoch_time,
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_student_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            logger.info("New best model saved (val_loss=%.4f)", val_loss)

    # Restore best model
    if best_student_state is not None:
        student.load_state_dict(best_student_state)
        student = student.cpu()

    logger.info("Distillation complete. Best val_loss=%.4f", best_val_loss)

    return {
        "best_val_loss": best_val_loss,
        "metrics": metrics,
        "student": student,
        "student_state_dict": best_student_state,
    }


def _validate(
    teacher: nn.Module,
    student: nn.Module,
    dataloader: Any,
    criterion: DistillationLoss,
    device: torch.device,
) -> float:
    """Run validation loop."""
    teacher.eval()
    student.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            teacher_actions = teacher.select_action(batch)
            student_actions = student.select_action(batch)

            loss = criterion(student_actions, teacher_actions)
            total_loss += loss.item()
            num_batches += 1

    student.train()
    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for distillation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Distill a LeRobot teacher policy into a smaller student"
    )
    parser.add_argument(
        "--teacher",
        type=str,
        required=True,
        help="Teacher policy checkpoint path or HuggingFace Hub ID",
    )
    parser.add_argument(
        "--student-config",
        type=str,
        required=True,
        help="Student model configuration (JSON file or preset name)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Training dataset path or HuggingFace Hub ID",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the distilled checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Training device")

    args = parser.parse_args()

    logger.info("Distillation CLI not yet fully implemented")
    logger.info("Teacher: %s, Student config: %s, Dataset: %s", args.teacher, args.student_config, args.dataset)


if __name__ == "__main__":
    main()
