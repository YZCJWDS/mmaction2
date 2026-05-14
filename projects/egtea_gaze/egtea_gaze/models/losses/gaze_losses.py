"""Loss functions for gaze supervision."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.registry import MODELS


def _prepare_pred_and_target(pred_attn: torch.Tensor,
                             target_gaze: torch.Tensor,
                             gaze_valid: torch.Tensor):
    if pred_attn.dim() == 5 and pred_attn.size(1) == 1:
        pred_attn = pred_attn[:, 0]
    pred_attn = pred_attn.float()
    target_gaze = target_gaze.float()
    gaze_valid = gaze_valid.float()

    if target_gaze.dim() == 5 and target_gaze.size(1) == 1:
        target_gaze = target_gaze[:, 0]
    if target_gaze.shape[1:] != pred_attn.shape[1:]:
        target_gaze = F.interpolate(
            target_gaze.unsqueeze(1),
            size=pred_attn.shape[1:],
            mode='trilinear',
            align_corners=False).squeeze(1)
    if gaze_valid.shape[1] != pred_attn.shape[1]:
        gaze_valid = F.interpolate(
            gaze_valid.unsqueeze(1).unsqueeze(-1),
            size=(pred_attn.shape[1], 1),
            mode='nearest').squeeze(-1).squeeze(1)
    return pred_attn, target_gaze, gaze_valid


@MODELS.register_module()
class GazeKLLoss(nn.Module):
    """KL loss between predicted attention logits and gaze targets."""

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self,
                pred_attn: torch.Tensor,
                target_gaze: torch.Tensor,
                gaze_valid: torch.Tensor) -> torch.Tensor:
        pred_attn, target_gaze, gaze_valid = _prepare_pred_and_target(
            pred_attn, target_gaze, gaze_valid)
        valid_mask = gaze_valid > 0.5
        if not valid_mask.any():
            return pred_attn.sum() * 0.0

        pred_log_prob = F.log_softmax(
            pred_attn.reshape(pred_attn.shape[0], pred_attn.shape[1], -1),
            dim=-1)
        target = target_gaze.reshape(target_gaze.shape[0], target_gaze.shape[1], -1)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        kl = target * (torch.log(target.clamp_min(self.eps)) - pred_log_prob)
        kl = kl.sum(dim=-1)
        loss = kl[valid_mask]
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss * self.loss_weight


@MODELS.register_module()
class GazeMSELoss(nn.Module):
    """MSE loss between predicted attention probabilities and gaze targets."""

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred_attn: torch.Tensor,
                target_gaze: torch.Tensor,
                gaze_valid: torch.Tensor) -> torch.Tensor:
        pred_attn, target_gaze, gaze_valid = _prepare_pred_and_target(
            pred_attn, target_gaze, gaze_valid)
        valid_mask = gaze_valid > 0.5
        if not valid_mask.any():
            return pred_attn.sum() * 0.0
        pred_prob = F.softmax(
            pred_attn.reshape(pred_attn.shape[0], pred_attn.shape[1], -1),
            dim=-1)
        target = target_gaze.reshape(target_gaze.shape[0], target_gaze.shape[1], -1)
        loss = (pred_prob - target).pow(2).mean(dim=-1)
        loss = loss[valid_mask]
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss * self.loss_weight


@MODELS.register_module()
class GazeBCELoss(nn.Module):
    """Binary cross entropy on normalized attention maps."""

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred_attn: torch.Tensor,
                target_gaze: torch.Tensor,
                gaze_valid: torch.Tensor) -> torch.Tensor:
        pred_attn, target_gaze, gaze_valid = _prepare_pred_and_target(
            pred_attn, target_gaze, gaze_valid)
        valid_mask = gaze_valid > 0.5
        if not valid_mask.any():
            return pred_attn.sum() * 0.0
        pred_prob = torch.sigmoid(pred_attn)
        target = torch.nan_to_num(target_gaze, nan=0.0, posinf=0.0, neginf=0.0)
        loss = F.binary_cross_entropy(pred_prob, target, reduction='none')
        loss = loss.flatten(2).mean(dim=-1)
        loss = loss[valid_mask]
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss * self.loss_weight

