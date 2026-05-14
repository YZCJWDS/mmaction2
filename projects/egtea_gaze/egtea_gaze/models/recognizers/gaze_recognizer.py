"""Thin recognizer wrapper for gaze-aware visualization hooks."""

from __future__ import annotations

from mmaction.models.recognizers import Recognizer3D
from mmaction.registry import MODELS


@MODELS.register_module()
class GazeRecognizer3D(Recognizer3D):
    """Minimal wrapper around Recognizer3D.

    Training and prediction stay identical to the upstream recognizer. The
    extra method is only used by visualization scripts when attention maps are
    requested explicitly.
    """

    def get_attention_maps(self, inputs, data_samples=None, test_mode: bool = True):
        feats, _ = self.extract_feat(inputs, data_samples=data_samples, test_mode=test_mode)
        if not hasattr(self.cls_head, 'forward_with_attention'):
            raise AttributeError('cls_head does not expose forward_with_attention')
        _, attn_logits = self.cls_head.forward_with_attention(feats)
        return attn_logits

