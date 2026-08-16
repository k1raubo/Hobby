# mini-JEPA
 
Minimal JEPA (Joint-Embedding Predictive Architecture) implementation on MNIST, for learning purposes.
 
## What it does
 
Trains a self-supervised model that learns to predict the embedding of a masked block of image patches from the visible context - without any labels. Labels are only used afterward, in a linear probe, to check whether the learned representations are any good.
 
## Architecture
 
- **PatchEmbed** - splits a 28×28 image into 7×7=49 patches of 4×4 pixels, projects each to a 64-dim vector.
- **block_mask** - picks one contiguous rectangular block of patches as the prediction target; the rest is context.
- **Encoder** - small Transformer, used twice:
  - `context_encoder` - trained by gradient descent, sees only the visible patches.
  - `target_encoder` - an EMA copy of `context_encoder` (no gradient), sees the full image.
- **Predictor** - small Transformer that takes the context representation plus mask tokens (for the masked positions) and predicts their embeddings.
- **variance_loss** - VICReg-style regularizer. Penalizes low variance of representations across the batch, to prevent the model from collapsing to a constant output.
Loss = `smooth_l1(prediction, target) + var_loss_weight * variance_loss(context)`
 
## Why the variance loss
 
Without it, the encoder can trivially minimize the prediction loss by mapping every image to (nearly) the same representation - the loss goes to ~0 but the model learns nothing useful. The variance term forces representations to actually differ across examples in the batch.
 

