"""Frozen bert-base-uncased text encoder for the P1 50-token grid."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from .grid import DEFAULT_TOKENIZER_PATH


class TextEncoder:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_TOKENIZER_PATH,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = AutoModel.from_pretrained(
            str(model_path), local_files_only=True
        ).to(self.device).eval()

    def encode(self, grid: dict) -> np.ndarray:
        """Return last_hidden_state with shape (50, 768), including specials/pads."""
        inputs = {
            name: torch.as_tensor(grid[name], dtype=torch.long, device=self.device).unsqueeze(0)
            for name in ("input_ids", "attention_mask", "token_type_ids")
        }
        with torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state[0]
        result = hidden.detach().cpu().numpy().astype(np.float32, copy=False)
        if result.shape != (50, 768):
            raise RuntimeError(f"BERT returned {result.shape}, expected (50, 768)")
        if not np.isfinite(result).all():
            raise RuntimeError("BERT output contains NaN or Inf")
        return result
