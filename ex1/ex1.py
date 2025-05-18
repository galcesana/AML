import os
import math
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from model import ConvVAE

# output directory
os.makedirs('section3_outputs', exist_ok=True)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data preparation (20k training subset)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
full_train = MNIST(root='./data', train=True, download=True, transform=transform)
targets = full_train.targets.numpy()
train_idx, _ = train_test_split(np.arange(len(targets)), train_size=20000, stratify=targets)
train_dataset = Subset(full_train, train_idx)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Fixed samples for reconstructions: 1 per class
def get_fixed_indices(dataset, labels, per_class=1):
    idx = []
    taken = {c: 0 for c in range(10)}
    for i, (_, y) in enumerate(dataset):
        if taken[int(y)] < per_class:
            idx.append(i)
            taken[int(y)] += 1
        if all(v >= per_class for v in taken.values()):
            break
    return idx

train_fixed = torch.stack([train_dataset[i][0] for i in get_fixed_indices(train_dataset, targets, 1)]).to(device)
test_fixed = torch.stack([test_dataset[i][0] for i in get_fixed_indices(test_dataset, test_dataset.targets.numpy(), 1)]).to(device)

check_epochs = [1,5,10,20,30]

# Q1: Amortized VAE
latent_dim = 200
vae = ConvVAE(latent_dim).to(device)
opt = optim.Adam(vae.parameters(), lr=1e-3)
epochs = 30
train_losses, val_losses = [], []
for epoch in range(1, epochs+1):
    vae.train()
    running_loss = 0.0
    for x,_ in train_loader:
        x = x.to(device)
        recon, mu, logvar = vae(x)
        mse = F.mse_loss(recon, x, reduction='mean')
        kl = 0.5 * torch.mean(mu.pow(2) + logvar.exp() - logvar - 1)
        loss = mse + kl
        opt.zero_grad()
        loss.backward()
        opt.step()
        running_loss += loss.item() * x.size(0)
    train_losses.append(running_loss / len(train_loader.dataset))
    # Validation
    vae.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x,_ in test_loader:
            x = x.to(device)
            recon, mu, logvar = vae(x)
            mse = F.mse_loss(recon, x, reduction='mean')
            kl = 0.5 * torch.mean(mu.pow(2) + logvar.exp() - logvar - 1)
            val_loss += (mse + kl).item() * x.size(0)
    val_losses.append(val_loss / len(test_loader.dataset))
    if epoch in check_epochs:
        torch.save(vae.state_dict(), f'section3_outputs/amort_vae_epoch{epoch}.pt')
    print(f'Epoch {epoch} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

# Plot Q1 losses
plt.figure(); plt.plot(range(1, epochs+1), train_losses, label='Train'); plt.plot(range(1, epochs+1), val_losses, label='Val'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.savefig('section3_outputs/q1_loss_curve.png')

# Q1 reconstructions
def plot_recon(model, epoch, fixed, name):
    state = torch.load(f'section3_outputs/amort_vae_epoch{epoch}.pt')
    vae.load_state_dict(state)
    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encode(fixed)
        z = vae.reparameterize(mu, logvar)
        recon = vae.decode(z).cpu()
    orig = fixed.cpu()
    plt.figure(figsize=(5,2))
    for i in range(len(orig)):
        plt.subplot(2,len(orig),i+1); plt.imshow(orig[i].squeeze(), cmap='gray'); plt.axis('off')
        plt.subplot(2,len(orig),len(orig)+i+1); plt.imshow(recon[i].squeeze(), cmap='gray'); plt.axis('off')
    plt.suptitle(f'{name} Recon Epoch {epoch}')
    plt.savefig(f'section3_outputs/{name.lower()}_recon_epoch{epoch}.png')
    plt.close()

for e in check_epochs:
    plot_recon(vae, e, train_fixed, 'Train')
    plot_recon(vae, e, test_fixed, 'Val')

# Q2: Sampling from prior
def plot_samples(model, epoch):
    state = torch.load(f'section3_outputs/amort_vae_epoch{epoch}.pt')
    vae.load_state_dict(state)
    vae.eval()
    z = torch.randn(10, latent_dim, device=device)
    with torch.no_grad(): samples = vae.decode(z).cpu()
    plt.figure(figsize=(5,1))
    for i in range(10): plt.subplot(1,10,i+1); plt.imshow(samples[i].squeeze(), cmap='gray'); plt.axis('off')
    plt.suptitle(f'Samples Epoch {epoch}')
    plt.savefig(f'section3_outputs/q2_samples_epoch{epoch}.png')
    plt.close()

for e in check_epochs:
    plot_samples(vae, e)

# Q3: Latent Optimization VAE
decoder = ConvVAE(latent_dim).to(device)
# freeze encoder
for p in decoder.encoder.parameters(): p.requires_grad = False
params = list(decoder.fc_decode.parameters()) + list(decoder.decoder.parameters())
opt_dec = optim.Adam(params, lr=1e-3)
n_samples = len(train_dataset)
mu_params = nn.Parameter(torch.zeros(n_samples, latent_dim, device=device))
logvar_params = nn.Parameter(torch.zeros(n_samples, latent_dim, device=device))
opt_lat = optim.SGD([mu_params, logvar_params], lr=1e-2)
latent_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
lo_losses = []
for epoch in range(1, epochs+1):
    decoder.train()
    total = 0
    for batch_idx, (x, _) in enumerate(latent_loader):
        idx = batch_idx*128 + torch.arange(len(x), device=device)
        x = x.to(device)
        mu_b = mu_params[idx]
        logvar_b = logvar_params[idx]
        std_b = torch.exp(0.5*logvar_b)
        z = mu_b + std_b * torch.randn_like(std_b)
        recon = decoder.decode(z)
        mse = F.mse_loss(recon, x, reduction='mean')
        kl = 0.5 * torch.mean(mu_b.pow(2) + logvar_b.exp() - logvar_b - 1)
        loss = mse + kl
        opt_dec.zero_grad(); opt_lat.zero_grad()
        loss.backward(); opt_dec.step(); opt_lat.step()
        total += loss.item()*x.size(0)
    lo_losses.append(total/ n_samples)
    if epoch in check_epochs:
        torch.save(decoder.state_dict(), f'section3_outputs/lo_decoder_epoch{epoch}.pt')
    print(f'LO Epoch {epoch} - Loss: {lo_losses[-1]:.4f}')

# Q3 reconstructions
for e in check_epochs:
    # train fixed recon
    state = torch.load(f'section3_outputs/lo_decoder_epoch{e}.pt')
    decoder.load_state_dict(state)
    decoder.eval()
    with torch.no_grad():
        idxs = torch.tensor(get_fixed_indices(train_dataset, targets,1), device=device)
        mu_b = mu_params[idxs]; logvar_b = logvar_params[idxs]
        z = mu_b + torch.exp(0.5*logvar_b)*torch.randn_like(mu_b)
        recon = decoder.decode(z).cpu()
    orig = train_fixed.cpu()
    plt.figure(figsize=(5,2))
    for i in range(len(orig)):
        plt.subplot(2,10,i+1); plt.imshow(orig[i].squeeze(), cmap='gray'); plt.axis('off')
        plt.subplot(2,10,10+i+1); plt.imshow(recon[i].squeeze(), cmap='gray'); plt.axis('off')
    plt.suptitle(f'LO Train Recon Epoch {e}')
    plt.savefig(f'section3_outputs/q3_lo_train_recon_epoch{e}.png'); plt.close()
    # sampling
    z_prior = torch.randn(10, latent_dim, device=device)
    with torch.no_grad(): samples = decoder.decode(z_prior).cpu()
    plt.figure(figsize=(5,1))
    for i in range(10): plt.subplot(1,10,i+1); plt.imshow(samples[i].squeeze(), cmap='gray'); plt.axis('off')
    plt.suptitle(f'LO Samples Epoch {e}')
    plt.savefig(f'section3_outputs/q3_lo_samples_epoch{e}.png'); plt.close()

# Q4: Log-probability estimation
def log_gaussian(x, mu, logvar):
    return -0.5 * ( ((x-mu)**2) / logvar.exp() + logvar + math.log(2*math.pi) )

sigma_p = 0.4
results = {'digit': [], 'set': [], 'logp': []}
# gather 5 train & 5 test per digit
for set_name, dataset, data_loader in [('train', train_dataset, None), ('test', test_dataset, test_loader)]:
    for digit in range(10):
        # collect 5 indices
        idxs = [i for i,(x,y) in enumerate(dataset) if y==digit][:5]
        for i in idxs:
            x, _ = dataset[i]
            x = x.unsqueeze(0).to(device)
            # get amortized mu/logvar
            mu, logvar = vae.encode(x)
            M=1000
            epsilon = torch.randn((1,M,latent_dim), device=device)
            std = torch.exp(0.5*logvar).unsqueeze(1)
            z = mu.unsqueeze(1) + std * epsilon
            # log p(z)
            log_pz = -0.5 * torch.sum(z**2 + math.log(2*math.pi), dim=2)
            # reconstructions
            recon = vae.decode(z.view(-1,latent_dim)).view(1,M,1,28,28)
            # log p(x|z)
            recon_diff = (x.unsqueeze(1) - recon)
            log_px_z = torch.sum(-0.5*((recon_diff**2)/(sigma_p**2) + math.log(2*math.pi*sigma_p**2)), dim=[2,3,4])
            # log q(z)
            log_qz = torch.sum(log_gaussian(z, mu.unsqueeze(1), logvar.unsqueeze(1)), dim=2)
            # log p(x)
            log_w = log_pz + log_px_z - log_qz
            log_px = torch.logsumexp(log_w, dim=1) - math.log(M)
            results['digit'].append(digit)
            results['set'].append(set_name)
            results['logp'].append(log_px.item())
            # plot one image
            if i == idxs[0]:
                plt.figure(); plt.imshow(x.cpu().squeeze(), cmap='gray'); plt.title(f'{set_name} digit {digit} logp={log_px.item():.2f}'); plt.axis('off')
                plt.savefig(f'section3_outputs/q4_{set_name}_digit{digit}.png'); plt.close()

# aggregate
import pandas as pd
df = pd.DataFrame(results)
avg_per_digit = df.groupby('digit')['logp'].mean()
avg_train = df[df.set=='train']['logp'].mean()
avg_test = df[df.set=='test']['logp'].mean()
print('Avg log-prob per digit:\n', avg_per_digit)
print(f'Avg log-prob train: {avg_train:.2f}, test: {avg_test:.2f}')

# Save averages
df.to_csv('section3_outputs/q4_logp_values.csv', index=False)
