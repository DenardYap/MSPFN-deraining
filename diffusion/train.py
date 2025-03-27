import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import trange
from torch.optim import Adam
from scipy.stats import gaussian_kde
from ddpm import UNet
from utils import *
torch.manual_seed(3)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet()
model.to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and std dev for MNIST
])

train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

n_epochs =   100
batch_size =  64
lr=5e-4
w = 0.5 # guidance weight
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

"""
(maybe TODO):
1. Add cosine scheduler
2. Add data augmentation like rotation and stuff 
"""

def train(timesteps, conditioned):
    cosine_beta_schedule(timesteps, s=0.008)

    optimizer = Adam(model.parameters(), lr=lr)
    tqdm_epoch = trange(n_epochs)
    for _ in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for images, labels in train_loader:
            images = images.to(device)
            t = torch.randint(0, timesteps, (images.shape[0],), device=device).long()

            if conditioned:
                loss = p_losses_condition(model, images, t, labels)
            else:
                loss = p_losses(model, images, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * images.shape[0]
            num_items += images.shape[0]
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        if conditioned:
            filename = f'q5_conditioned_accurate_100_{timesteps}.pth'
        else:
            filename = f'q5_{timesteps}.pth'

        torch.save(model.state_dict(), filename)
    print(f"timesteps : {timesteps} avg_loss is: {avg_loss} conditioned?: {conditioned}")

if __name__=="__main__":
    for timesteps in [1024]:
        # train(timesteps, False)
        train(timesteps, True)