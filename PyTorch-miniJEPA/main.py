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


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(28 * 28, 512)
        self.linear2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x


class Trainer:
    def __init__(self, model, optimizer, criteria):
        self.model = model
        self.optimizer = optimizer
        self.criteria = criteria

    def train(self, dataloader, num_epoch):
        self.model.train()
        
        for epoch in range(num_epoch):
            pbar = tqdm.tqdm(dataloader, desc = f"Epoch {epoch}/{num_epoch}")
            total_loss = 0

            for idx, batch in enumerate(pbar):
                inputs = batch["image"].float()
                outputs = batch["label"]

                self.optimizer.zero_grad()
                predictions = self.model(inputs)
                loss = self.criteria(predictions, outputs)

                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                if idx > 0:
                    pbar.set_postfix(loss=f"{total_loss / idx}")
        

model = NeuralNetwork().to(device)
optimier = optim.AdamW(model.parameters(), lr=0.0001)
criteria = nn.CrossEntropyLoss()
trainer = Trainer(model, optimier, criteria)
trainer.train(train_loader, 5)


metric = evaluate.load("accuracy")
model.eval()
pbar = tqdm.tqdm(test_loader)

with torch.no_grad():
    for batch in pbar:
        inputs = batch["image"].float()
        outputs = batch["label"]

        predictions = model(inputs).argmax(dim=-1)
        metric.add_batch(predictions=predictions, references=outputs)

print(metric.compute())