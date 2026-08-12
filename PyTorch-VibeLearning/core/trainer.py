from typing import Any, Callable

import torch
import torch.nn as nn


class VibeTrainer:
    def __init__(
        self,
        model: nn.Module,
        metric=None,
        context_samples: int | None = None,
        verbose: bool = True,
    ):
        self.model = model
        self.metric = metric
        self.context_samples = context_samples
        self.verbose = verbose
        self.history: list[dict] = []

    def fit(
        self,
        inputs: Any,
        outputs: Any,
        epochs: int = 10,
        eval_inputs: Any | None = None,
        eval_targets: Any | None = None,
    ) -> list[dict]:
        inputs_t   = _as_float(inputs)
        outputs_t  = _as_target(outputs)
        has_eval   = eval_inputs is not None and eval_targets is not None
        eval_in_t  = _as_float(eval_inputs) if has_eval else inputs_t
        eval_tgt   = eval_targets if has_eval else outputs
        eval_tgt_t = _as_target(eval_tgt)
        ctx_in, ctx_out = _context(inputs_t, outputs_t, self.context_samples)

        for epoch in range(1, epochs + 1):
            self.model.vibe_step(ctx_in, ctx_out)

            row = {}
            if self.metric is not None:
                with torch.no_grad():
                    preds = self.model(eval_in_t)
                refs = eval_tgt if isinstance(eval_tgt, list) else eval_tgt_t.long().tolist()
                row.update(self.metric.compute(predictions=preds.argmax(dim=1).tolist(), references=refs))

            self.history.append(row)
            if self.verbose:
                _log(epoch, epochs, row)

        return self.history

    def fit_loader(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 10,
        eval_loader: torch.utils.data.DataLoader | None = None,
    ) -> list[dict]:
        for epoch in range(1, epochs + 1):
            for batch_in, batch_out in dataloader:
                ctx_in, ctx_out = _context(batch_in, batch_out, self.context_samples)
                self.model.vibe_step(ctx_in, ctx_out)

            row = {}
            if self.metric is not None:
                src = eval_loader or dataloader
                all_preds, all_refs = [], []
                with torch.no_grad():
                    for batch_in, batch_out in src:
                        all_preds.extend(self.model(batch_in).argmax(dim=1).tolist())
                        all_refs.extend(batch_out.long().tolist())
                row.update(self.metric.compute(predictions=all_preds, references=all_refs))

            self.history.append(row)
            if self.verbose:
                _log(epoch, epochs, row)

        return self.history


def _as_float(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.float()
    return torch.tensor(value, dtype=torch.float32)


def _as_target(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    t = torch.tensor(value)
    return t.long() if not t.is_floating_point() else t.float()


def _context(inputs: torch.Tensor, outputs: torch.Tensor, n: int | None):
    if n is None or len(inputs) <= n:
        return inputs, outputs
    idx = torch.randperm(len(inputs))[:n]
    return inputs[idx], outputs[idx]


def _log(epoch: int, total: int, row: dict) -> None:
    w = len(str(total))
    parts = [f"Epoch {epoch:{w}}/{total}"] + [f"{k}={v:.4f}" for k, v in row.items()]
    print("  ".join(parts))
