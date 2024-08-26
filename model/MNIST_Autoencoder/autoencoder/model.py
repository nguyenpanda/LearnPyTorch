import torch
from torch import nn
from nguyenpanda.crow import nb_utils

IN_COLAB = nb_utils.is_colab()

class MNISTAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.train_call = 0
        self.zip = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        )

        self.unzip = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.zip(x)
        return self.unzip(x)

    def test_step(self, dataloader, loss_function):
        total_batch = len(dataloader)
        total_loss = 0

        self.eval()
        with torch.inference_mode():
            for b, (x, y) in enumerate(dataloader, 1):
                logit = self.forward(x)
                loss = loss_function(logit, y)

                total_loss += loss.item()
        return total_loss / total_batch

    def train_step(self, dataloader, loss_function, optimizer):
        self.train_call += 1
        total_batch = len(dataloader)
        b_pad = len(str(total_batch))
        total_loss = 0

        self.train()
        for b, (x, y) in enumerate(dataloader, 1):
            logit = self.forward(x)

            loss = loss_function(logit, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            text = '{: >{}}/{} | ins_loss = \033[1;92m{: >10.5f}\033[0m'
            if IN_COLAB:
                text = '\r' + text
            else:
                text = text + '\r'
            print(text.format(b, b_pad, total_batch, loss.item()), end='')
        print()

        return total_loss / total_batch