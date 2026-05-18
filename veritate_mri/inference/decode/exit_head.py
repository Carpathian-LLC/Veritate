# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - ExitHead: a tiny classifier that predicts "should we exit at this layer".
#   Replaces the expensive logit-lens projection used by Brain.stream_fast's
#   adaptive-depth mode.
# - The smoke (S54) found that projecting each layer's residual through
#   final_norm + tied LM head on the 800M (28 layers, 803M params) costs
#   more compute than it saves. A 50K-param Linear(hidden, 1) sigmoid
#   classifier costs orders of magnitude less and can be trained on the
#   target model's own per-layer confidence pattern.
# - Training (offline, in experiments/v2/exit_head/): feed N text chunks
#   through the FROZEN target model, capture each layer's residual at the
#   last position + the FINAL-layer top-1 confidence as the label. Train
#   the per-layer classifier to predict "the final argmax will equal this
#   layer's argmax." Loss = BCE on that target.
# - At inference, Brain.stream_fast(mode='adaptive', exit_head=...) uses
#   exit_head.predict(residual, layer) -> bool. Far cheaper than the
#   logit-lens approach (one matmul of size [1, hidden] -> [1] per layer
#   instead of [1, hidden] -> [1, vocab=256]).
# - This is the LayerSkip / CALM-style design. The model class is shipped
#   here in veritate_mri/decode/; the training script lives in
#   experiments/v2/exit_head/.
# veritate_mri/decode/exit_head.py
# ------------------------------------------------------------------------------------
# Imports:

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------------
# Constants

# ------------------------------------------------------------------------------------
# Functions


class ExitHead(nn.Module):
    """One classifier per intermediate layer. Predicts whether the final-layer
    argmax will equal this layer's argmax, given the current residual."""

    def __init__(self, hidden: int, n_layers: int, dropout: float = 0.0):
        super().__init__()
        self.hidden   = hidden
        self.n_layers = n_layers
        # Per-layer linear classifier. Light. Total params = n_layers * (hidden + 1).
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden, 1) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @torch.no_grad()
    def predict_exit(self, residual_last_pos: torch.Tensor, layer_idx: int) -> float:
        """Return probability in [0, 1] that exiting at layer_idx is safe.
        `residual_last_pos`: [hidden]. Returns a Python float."""
        x = self.dropout(residual_last_pos)
        logit = self.classifiers[layer_idx](x.unsqueeze(0))   # [1, 1]
        return float(torch.sigmoid(logit).item())

    def forward(self, residuals_per_layer: List[torch.Tensor]) -> torch.Tensor:
        """Training-time forward. Returns [n_layers] logits.
        `residuals_per_layer`: list of [B, hidden] tensors (one per layer)."""
        outs = []
        for L, r in enumerate(residuals_per_layer):
            r = self.dropout(r)
            outs.append(self.classifiers[L](r).squeeze(-1))    # [B]
        return torch.stack(outs, dim=-1)                       # [B, n_layers]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def make_random_exit_head(target_model) -> ExitHead:
    """Build an untrained ExitHead matching target_model's shape. Useful for
    unit tests + as a starting point for training."""
    return ExitHead(hidden=target_model.hidden, n_layers=target_model.layers)
