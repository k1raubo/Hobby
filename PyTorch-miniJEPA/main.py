import copy
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("ylecun/mnist")
dataset.set_format(type="torch", columns=["image", "label"], device=device)
train_loader = DataLoader(dataset["train"], batch_size=64, shuffle=True)
test_loader = DataLoader(dataset["test"], batch_size=64, shuffle=False)

PATCH_SIZE = 4
IMG_SIZE = 28
GRID_SIDE = IMG_SIZE // PATCH_SIZE
NUM_PATCHES = GRID_SIDE * GRID_SIDE
EMBED_DIM = 64


class PatchEmbed(nn.Module):
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=1, embed_dim=EMBED_DIM):
        super().__init__()
        self.grid_side = img_size // patch_size
        self.num_patches = self.grid_side ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def block_mask(grid_side, batch_size, mask_ratio=0.5, device="cpu"):
    N = grid_side
    total = N * N
    target_area = max(1, int(total * mask_ratio))
    side = max(1, min(N, int(target_area ** 0.5)))

    ctx_list, tgt_list = [], []
    grid = torch.arange(total).reshape(N, N)
    for _ in range(batch_size):
        top = torch.randint(0, N - side + 1, (1,)).item()
        left = torch.randint(0, N - side + 1, (1,)).item()

        tgt_block = grid[top:top + side, left:left + side].reshape(-1)
        tgt_set = set(tgt_block.tolist())
        ctx_block = torch.tensor([i for i in range(total) if i not in tgt_set])

        ctx_list.append(ctx_block)
        tgt_list.append(tgt_block)

    context_indices = torch.stack(ctx_list).to(device)
    target_indices = torch.stack(tgt_list).to(device)
    return context_indices, target_indices


def variance_loss(repr, eps=1e-4, target_std=1.0):
    if repr.dim() == 3:
        repr = repr.reshape(-1, repr.shape[-1])
    var = repr.var(dim=0).clamp(min=eps)
    std = torch.sqrt(var)
    return F.relu(target_std - std).mean()


class Encoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, depth=3, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.blocks(x))


class Predictor(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, pred_dim=32, depth=2, num_heads=4):
        super().__init__()
        self.embed_to_pred = nn.Linear(embed_dim, pred_dim)
        self.pos_to_pred = nn.Linear(embed_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=pred_dim,
            nhead=num_heads,
            dim_feedforward=pred_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(pred_dim)
        self.pred_to_embed = nn.Linear(pred_dim, embed_dim)

    def forward(self, context_tokens, pos_embed_full, target_indices):
        B = context_tokens.shape[0]
        ctx = self.embed_to_pred(context_tokens)

        N_tgt = target_indices.shape[1]
        query_pos = torch.gather(
            pos_embed_full.expand(B, -1, -1), 1,
            target_indices.unsqueeze(-1).expand(-1, -1, pos_embed_full.shape[-1])
        )
        query_pos = self.pos_to_pred(query_pos)
        query_tokens = self.mask_token.expand(B, N_tgt, -1) + query_pos

        seq = torch.cat([ctx, query_tokens], dim=1)
        seq = self.norm(self.blocks(seq))

        pred_targets = seq[:, -N_tgt:, :]
        return self.pred_to_embed(pred_targets)


class JEPA(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, ema_decay=0.996, mask_ratio=0.5, var_loss_weight=1.0):
        super().__init__()
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, NUM_PATCHES, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.context_encoder = Encoder(embed_dim=embed_dim)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = Predictor(embed_dim=embed_dim)
        self.ema_decay = ema_decay
        self.mask_ratio = mask_ratio
        self.var_loss_weight = var_loss_weight

    @torch.no_grad()
    def update_target_encoder(self):
        for tp, cp in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            tp.data.mul_(self.ema_decay).add_(cp.data, alpha=1 - self.ema_decay)

    def forward(self, images):
        B, device = images.shape[0], images.device

        tokens = self.patch_embed(images) + self.pos_embed
        ctx_idx, tgt_idx = block_mask(GRID_SIDE, B, self.mask_ratio, device=device)

        ctx_tokens = torch.gather(tokens, 1, ctx_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        ctx_repr = self.context_encoder(ctx_tokens)

        with torch.no_grad():
            full_repr = self.target_encoder(tokens)
            target_repr = torch.gather(full_repr, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, full_repr.shape[-1]))

        pred_repr = self.predictor(ctx_repr, self.pos_embed, tgt_idx)

        pred_loss = F.smooth_l1_loss(pred_repr, target_repr)
        var_loss = variance_loss(ctx_repr)
        loss = pred_loss + self.var_loss_weight * var_loss
        return loss, pred_loss, var_loss

    @torch.no_grad()
    def encode_full_image(self, images):
        tokens = self.patch_embed(images) + self.pos_embed
        return self.context_encoder(tokens).mean(dim=1)


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train(self, dataloader, num_epoch):
        self.model.train()
        for epoch in range(num_epoch):
            pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch}/{num_epoch}")
            total_loss, total_pred, total_var = 0, 0, 0
            for idx, batch in enumerate(pbar):
                inputs = batch["image"].float() / 255.0
                if inputs.dim() == 3:
                    inputs = inputs.unsqueeze(1)

                self.optimizer.zero_grad()
                loss, pred_loss, var_loss = self.model(inputs)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nskipping batch {idx} (epoch {epoch}): loss={loss.item()}")
                    continue

                total_loss += loss.item()
                total_pred += pred_loss.item()
                total_var += var_loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.model.update_target_encoder()

                if idx > 0:
                    pbar.set_postfix(
                        loss=f"{total_loss / (idx + 1):.4f}",
                        pred=f"{total_pred / (idx + 1):.4f}",
                        var=f"{total_var / (idx + 1):.4f}",
                    )


model = JEPA().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
trainer = Trainer(model, optimizer)
trainer.train(train_loader, num_epoch=5)


class LinearProbe(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


probe = LinearProbe().to(device)
probe_optimizer = optim.AdamW(probe.parameters(), lr=1e-3)
criteria = nn.CrossEntropyLoss()

model.eval()
for epoch in range(3):
    pbar = tqdm.tqdm(train_loader, desc=f"Probe epoch {epoch}/3")
    total_loss = 0
    for idx, batch in enumerate(pbar):
        inputs = batch["image"].float() / 255.0
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)
        labels = batch["label"]

        with torch.no_grad():
            features = model.encode_full_image(inputs)

        probe_optimizer.zero_grad()
        logits = probe(features)
        loss = criteria(logits, labels)
        total_loss += loss.item()
        loss.backward()
        probe_optimizer.step()

        if idx > 0:
            pbar.set_postfix(loss=f"{total_loss / (idx + 1):.4f}")


metric = evaluate.load("accuracy")
probe.eval()
pbar = tqdm.tqdm(test_loader)
with torch.no_grad():
    for batch in pbar:
        inputs = batch["image"].float() / 255.0
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)
        labels = batch["label"]

        features = model.encode_full_image(inputs)
        predictions = probe(features).argmax(dim=-1)
        metric.add_batch(predictions=predictions, references=labels)

print(metric.compute())
