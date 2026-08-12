# PyTorch VibeLearning

Train PyTorch models using an LLM as the optimizer instead of backpropagation.

Each training step sends the current weights, a batch of inputs, expected outputs, and the model's current predictions to GPT-4o. The LLM reasons about the error and returns updated weights.

> **SPOILER**
>
> It doesn't work lol.

## How it works

`@vibelearn` is a class decorator that adds a `vibe_step(inputs, outputs)` method to any `nn.Module`. Calling it:

1. Runs a forward pass to get current predictions.
2. Serializes the weights, inputs, expected outputs, and predictions as JSON.
3. Sends them to GPT-4o with a structured output schema (`UpdatedWeights`).
4. Writes the returned weights back into the model in-place.

`VibeTrainer` wraps this in a standard training loop with per-epoch loss and metric logging.