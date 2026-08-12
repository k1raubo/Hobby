import functools
from typing import Any

import torch
import torch.nn as nn
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from core.prompt import INSTRUCTIONS, ArchitectureInfo, LayerInfo

load_dotenv()


class UpdatedLayer(BaseModel):
    name: str = Field(description="Layer parameter name, e.g. 'fc1.weight'.")
    weights: list[float] = Field(description="Updated flat weight list.")


class UpdatedWeights(BaseModel):
    layers: list[UpdatedLayer] = Field(
        description="Updated weights for every layer, in the same order as the input."
    )


def _to_python(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    return value


def vibelearn(cls):
    original_init = cls.__init__

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._vibe_client = OpenAI()

    def vibe_step(self, inputs: Any, outputs: Any) -> UpdatedWeights:
        inputs_t = inputs if isinstance(inputs, torch.Tensor) else torch.tensor(inputs, dtype=torch.float32)
        outputs_t = outputs if isinstance(outputs, torch.Tensor) else torch.tensor(outputs)

        with torch.no_grad():
            preds = self(inputs_t)

        info = ArchitectureInfo(
            architecture=cls.__name__,
            layers={
                name: LayerInfo(weights=param.data.flatten().tolist())
                for name, param in self.named_parameters()
            },
            inputs=_to_python(inputs_t),
            expected_outputs=_to_python(outputs_t),
            current_predictions=_to_python(preds),
        )

        completion = self._vibe_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": info.model_dump_json(indent=2)},
            ],
            response_format=UpdatedWeights,
        )

        updated: UpdatedWeights = completion.choices[0].message.parsed
        layer_map = {ul.name: ul.weights for ul in updated.layers}

        with torch.no_grad():
            for name, param in self.named_parameters():
                if name not in layer_map:
                    continue
                param.copy_(
                    torch.tensor(layer_map[name], dtype=param.dtype).reshape(param.shape)
                )

        return updated

    cls.__init__ = __init__
    cls.vibe_step = vibe_step
    return cls
