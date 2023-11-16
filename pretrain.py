import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ebm import GraphormerEBM
from sampling import langevin_sampling, hmc_sampling
from torch.utils.data import DataLoader

import wandb

wandb.init(project='ebm_pretraining', entity='andrewliu1324')


class Trainer():
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self, train_loader, valid_loader, n_epochs, n_steps, step_size, n_samples, verbose=True):
        """ Train the model. """
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader, n_steps, step_size, n_samples)
            valid_loss = self.evaluate(valid_loader)

            wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

            if verbose:
                print(f'Epoch {epoch + 1} | Train loss: {train_loss:.3f} | Valid loss: {valid_loss:.3f}')

            self.scheduler.step()

    def train_epoch(self, train_loader, n_steps, step_size, n_samples):
        """ Train the model for one epoch. """
        self.model.train()
        train_loss = 0

        for batch in train_loader:
            batch = batch.to(self.device)
            loss = self.model.energy(batch).mean()
            loss.backward()

            # Perform Langevin sampling
            with torch.no_grad():
                x = langevin_sampling(self.model, batch, step_size, n_steps)

            # Perform HMC sampling
            #with torch.no_grad():
            #    x = hmc_sampling(self.model.energy, batch, step_size, n_steps, n_samples)

            # Update parameters
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()

        return train_loss / len(train_loader)

    def evaluate(self, data_loader):
        """ Evaluate the model. """
        self.model.eval()
        loss = 0

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                loss += self.model.energy(batch).mean().item()

        return loss / len(data_loader)

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    