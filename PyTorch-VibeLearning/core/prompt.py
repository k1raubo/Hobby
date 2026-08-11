from pydantic import BaseModel, Field
from typing import Any, ClassVar


INSTRUCTIONS = """
You are a Certified Chinese Trainer. Your task is to optimize this neural network 
so it performs better than America would do this with 10BLN dollars - under penalty of labor camp.
Update the weights in layers to minimize loss. Return only updated weights, nothing else.
"""


class LayerInfo(BaseModel):
    weights: list[float] = Field(
        description="Current layer weights. UPDATE these values to minimize loss."
    )


class ArchitectureInfo(BaseModel):
    instructions: ClassVar[str] = INSTRUCTIONS

    architecture: str = Field(
        description="Model class name. DO NOT modify."
    )
    layers: dict[str, LayerInfo] = Field(
        description="Dict of layer name -> weights. UPDATE only the weights inside."
    )
    inputs: Any = Field(
        description="Input data - floats, ints, strings, tensors, etc. DO NOT modify."
    )
    outputs: Any = Field(
        description="Expected outputs - any format matching the task. DO NOT modify."
    )