"""Teacher-student distillation for LeRobot policies."""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from lerobot_edge.core.base import NativePyTorchBackend
from lerobot_edge.core.configs import EdgeBaseConfig

logger = logging.getLogger(__name__)

__all__ = [
    "DistillationLoss",
    "DistilledBackend",
    "distill",
]


class DistillationLoss(nn.Module):
    """Combined MSE + KL divergence loss for knowledge distillation."""

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
        mse_loss = F.mse_loss(student_actions, teacher_actions)

        if student_logits is not None and teacher_logits is not None:
            student_log_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
            kl_loss = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (
                self.temperature**2
            )
            return (1 - self.alpha) * mse_loss + self.alpha * kl_loss
        else:
            return mse_loss


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
    """Run knowledge distillation training."""
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

    teacher = teacher.to(dev).eval()
    student = student.to(dev).train()

    for param in teacher.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = DistillationLoss(temperature=temp, alpha=a)

    best_val_loss = float("inf")
    best_student_state = None
    metrics: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "learning_rate": []}

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_dataloader:
            batch = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with torch.no_grad():
                teacher_actions = teacher.select_action(batch)

            student_actions = student.select_action(batch)
            loss = criterion(student_actions, teacher_actions)

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_student_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            logger.info("New best model saved (val_loss=%.4f)", val_loss)

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
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }

            teacher_actions = teacher.select_action(batch)
            student_actions = student.select_action(batch)

            loss = criterion(student_actions, teacher_actions)
            total_loss += loss.item()
            num_batches += 1

    student.train()
    return total_loss / max(num_batches, 1)


def main() -> None:
    """CLI entry point for distillation."""
    raise SystemExit(
        "lerobot-edge-distill: not yet implemented.\n"
        "Use the distill() function directly for distillation."
    )


if __name__ == "__main__":
    main()
