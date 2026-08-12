from pydantic import BaseModel, Field
from typing import Any, ClassVar


INSTRUCTIONS = """
You are an expert neural network optimizer performing manual gradient descent.

You will receive:
- The current weights of each layer (flat lists)
- A batch of input samples and their expected outputs
- The model's current predictions on those inputs

Your job: update the weights to REDUCE the loss.

How to reason about weight updates:
- Compare predictions to expected outputs to understand the error direction
- For each layer, nudge weights in the direction that would reduce the output error
- Use small updates (magnitude 0.01 - 0.1) -- large jumps overshoot
- Bias terms (*.bias) shift outputs directly; weight matrices scale features
- If predictions are too high for a class, reduce weights feeding into it
- If predictions are too low for a class, increase weights feeding into it

Return ALL layers with updated weights. Keep the same number of weights per layer.
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
        description="Layer name -> current weights. UPDATE only the weight values."
    )
    inputs: Any = Field(
        description="Input samples fed to the network. DO NOT modify."
    )
    expected_outputs: Any = Field(
        description="Ground-truth target labels/values. DO NOT modify."
    )
    current_predictions: Any = Field(
        description="What the model currently predicts for these inputs. Use this to judge the error."
    )
