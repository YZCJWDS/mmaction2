"""SlowFast head with lightweight gaze supervision."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model.weight_init import normal_init

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, SampleList
from mmaction.models.heads.base import BaseHead


@MODELS.register_module()
class GazeSlowFastHead(BaseHead):
    """Action classification head with an auxiliary gaze attention branch."""

    def __init__(self,
                 num_classes: int,
                 in_channels: Sequence[int] = (2048, 256),
                 mid_channels: int = 256,
                 spatial_type: str = 'avg',
                 dropout_ratio: float = 0.5,
                 init_std: float = 0.01,
                 gaze_loss: ConfigType = dict(type='GazeKLLoss', loss_weight=0.1),
                 attention_size: Optional[Tuple[int, int]] = (14, 14),
                 temporal_size: Optional[int] = None,
                 loss_cls: ConfigType = dict(
                     type='CrossEntropyLoss', loss_weight=1.0),
                 average_clips: Optional[str] = 'prob',
                 **kwargs) -> None:
        self.branch_channels = tuple(int(v) for v in in_channels)
        super().__init__(
            num_classes=num_classes,
            in_channels=sum(self.branch_channels),
            loss_cls=loss_cls,
            average_clips=average_clips,
            **kwargs)
        self.mid_channels = int(mid_channels)
        self.spatial_type = spatial_type
        self.dropout_ratio = float(dropout_ratio)
        self.init_std = float(init_std)
        self.attention_size = attention_size
        self.temporal_size = temporal_size
        self.gaze_loss = MODELS.build(gaze_loss)

        slow_channels = max(1, self.mid_channels // 2)
        fast_channels = max(1, self.mid_channels - slow_channels)
        self.slow_proj = nn.Conv3d(self.branch_channels[0], slow_channels, kernel_size=1)
        self.fast_proj = nn.Conv3d(self.branch_channels[1], fast_channels, kernel_size=1)
        self.attn_fuse = nn.Conv3d(self.mid_channels, self.mid_channels, kernel_size=1)
        self.attn_pred = nn.Conv3d(self.mid_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1)) if spatial_type == 'avg' else None
        self.dropout = nn.Dropout(p=self.dropout_ratio) if self.dropout_ratio > 0 else None
        self.fc_cls = nn.Linear(sum(self.branch_channels), num_classes)

    def init_weights(self) -> None:
        normal_init(self.fc_cls, std=self.init_std)
        normal_init(self.slow_proj, std=self.init_std)
        normal_init(self.fast_proj, std=self.init_std)
        normal_init(self.attn_fuse, std=self.init_std)
        normal_init(self.attn_pred, std=self.init_std)

    def _forward_cls(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_slow, x_fast = x
        if self.avg_pool is not None:
            x_slow = self.avg_pool(x_slow)
            x_fast = self.avg_pool(x_fast)
        cls_feat = torch.cat((x_fast, x_slow), dim=1)
        if self.dropout is not None:
            cls_feat = self.dropout(cls_feat)
        return self.fc_cls(cls_feat.view(cls_feat.size(0), -1))

    def _forward_attention(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_slow, x_fast = x
        fast_aligned = F.interpolate(
            x_fast,
            size=x_slow.shape[2:],
            mode='trilinear',
            align_corners=False)
        fused = torch.cat((self.slow_proj(x_slow), self.fast_proj(fast_aligned)), dim=1)
        fused = self.relu(self.attn_fuse(fused))
        if self.attention_size is not None:
            target_t = self.temporal_size or fused.shape[2]
            fused = F.interpolate(
                fused,
                size=(target_t, self.attention_size[0], self.attention_size[1]),
                mode='trilinear',
                align_corners=False)
        attn_logits = self.attn_pred(fused).squeeze(1)
        return attn_logits

    def forward_with_attention(self,
                               x: Tuple[torch.Tensor, torch.Tensor]
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_score = self._forward_cls(x)
        attn_logits = self._forward_attention(x)
        return cls_score, attn_logits

    def forward(self,
                x: Tuple[torch.Tensor, torch.Tensor],
                return_attention: bool = False,
                **kwargs):
        cls_score, attn_logits = self.forward_with_attention(x)
        if return_attention:
            return cls_score, attn_logits
        return cls_score

    def _collect_gaze_targets(self,
                              data_samples: SampleList,
                              device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        template_maps = None
        template_valid = None
        maps_list = []
        valid_list = []
        has_any = False
        for data_sample in data_samples:
            gaze_maps = getattr(data_sample, 'gaze_maps', None)
            gaze_valid = getattr(data_sample, 'gaze_valid', None)
            if gaze_maps is not None and gaze_valid is not None:
                gaze_maps = torch.as_tensor(gaze_maps, device=device)
                gaze_valid = torch.as_tensor(gaze_valid, device=device)
                template_maps = gaze_maps if template_maps is None else template_maps
                template_valid = gaze_valid if template_valid is None else template_valid
                has_any = has_any or bool(gaze_valid.float().sum() > 0)
            maps_list.append(gaze_maps)
            valid_list.append(gaze_valid)

        if template_maps is None or template_valid is None:
            return None, None

        stacked_maps = []
        stacked_valid = []
        for gaze_maps, gaze_valid in zip(maps_list, valid_list):
            if gaze_maps is None or gaze_valid is None:
                stacked_maps.append(torch.zeros_like(template_maps))
                stacked_valid.append(torch.zeros_like(template_valid))
            else:
                stacked_maps.append(gaze_maps.float())
                stacked_valid.append(gaze_valid.float())
        if not has_any:
            return torch.stack(stacked_maps), torch.stack(stacked_valid)
        return torch.stack(stacked_maps), torch.stack(stacked_valid)

    def loss(self,
             feats: Tuple[torch.Tensor, torch.Tensor],
             data_samples: SampleList,
             **kwargs):
        cls_score, attn_logits = self.forward_with_attention(feats)
        losses = self.loss_by_feat(cls_score, data_samples)
        target_gaze, gaze_valid = self._collect_gaze_targets(data_samples, cls_score.device)
        if target_gaze is None or gaze_valid is None:
            losses['loss_gaze'] = attn_logits.sum() * 0.0
        else:
            losses['loss_gaze'] = self.gaze_loss(attn_logits, target_gaze, gaze_valid)
        return losses

    def predict(self, feats, data_samples: SampleList, **kwargs) -> SampleList:
        cls_score = self.forward(feats)
        return self.predict_by_feat(cls_score, data_samples)

