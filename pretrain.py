import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sampling import sample_ebm, convert_to_data
from torch_geometric.loader import DataLoader
from dataset import gen_data
from torch_geometric.data import Data
from torch.nn.utils import clip_grad_value_
from collections import deque
from sat.models import GraphTransformer
import wandb
import os



class Trainer():
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self, train_loader, n_epochs, n_steps, step_size, clip_grad, verbose=True):
        """ Train the model. """
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader, n_steps, step_size, clip_grad)

            if verbose:
                print(f'Epoch {epoch + 1} | Train loss: {train_loss:.3f}')

            if self.scheduler:
                self.scheduler.step()

    def train_epoch(self, train_loader, n_steps, step_size, clip_grad):
        """ Train the model for one epoch. """
        self.model.train()
        train_loss = 0

        # Create a replay buffer
        replay_buffer_A = deque(maxlen = 10000)
        replay_buffer_X = deque(maxlen = 10000)
        
        for i, batch in enumerate(train_loader):

            # Sample negative pairs
            if np.random.rand() < 0.95 and len(replay_buffer_A) > 0 and len(replay_buffer_X) > 0:
                init_A = replay_buffer_A[np.random.randint(len(replay_buffer_A))].to(self.device)
                init_X = replay_buffer_X[np.random.randint(len(replay_buffer_X))].to(self.device)
            else:
                init_A = torch.rand(len(batch), 38, 38, 3).to(self.device)
                init_X = torch.rand(len(batch), 38, 9).to(self.device)
            self_A, self_X = sample_ebm(self.model, init_A, init_X, step_size=step_size, max_iterations=n_steps)
            # Update the replay buffer
            replay_buffer_A.append(self_A)
            replay_buffer_X.append(self_X)

            batch = batch.to(self.device)
            energy1 = self.model(batch)

            self_data_list = convert_to_data(self_A, self_X)
            self_loader = DataLoader(self_data_list, shuffle=False, batch_size=len(self_data_list))
            for self_batch in self_loader:
                self_batch = self_batch.to(self.device)
                energy2 = self.model(self_batch)
            loss1 = energy1.mean()
            loss2 = -energy2.mean()
            reg = (energy1**2).mean() + (energy2**2).mean()
            loss =  loss1 + loss2 + reg
            loss.backward()

            if i % 10 == 0:
                wandb.log({'Train loss': loss.item()})

            

            # Update parameters
            clip_grad_value_(self.model.parameters(), clip_grad)
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()

        return train_loss / len(train_loader)


def run_experiment():
    wandb.init(project='ebm_pretraining', entity='andrewliu1324')
    config = wandb.config

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = gen_data('/home/andrew/rl-ebm/data')
    dataloader = DataLoader(dataset, batch_size=128, num_workers=8, shuffle=True, pin_memory=True)
    
    model = GraphTransformer(num_layers=config.num_layers, num_heads=config.num_heads, d_model=config.d_model, batch_norm=config.batch_norm).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.0, 0.999))
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    trainer = Trainer(model, optimizer, None, device)
    trainer.train(dataloader, n_epochs=10, n_steps=config.n_steps, step_size=config.step_size, clip_grad=config.clip_grad, verbose=True)

    len_dir = len(os.listdir('/home/andrew/rl-ebm/checkpoints/'))
    torch.save(model.state_dict(), f'/home/andrew/rl-ebm/checkpoints/model_{len_dir}.pt')
def main():

    #Set up wandb sweep
    sweep_config = {
        'method': 'bayes',  # Can be grid, random, or bayes
        'metric': {
            'name': 'loss',
            'goal': 'minimize'   
        },
        'parameters': {
            'learning_rate': {
            'values': [0.0001]
            },
            'batch_size': {
            'values': [128]
            },
            'n_steps': { #K
                'values': [30]
            },
            'step_size': { #lambda
                'min': 20,
                'max': 100
            },
            'clip_grad':{
                'values':[0.01]
            },
            'batch_norm':{
                'values':[True, False]
            },
            'num_layers':{
                'values':[4]
            },
            'num_heads':{
                'values': [8]
            },
            'd_model':
            {
                'values': [64]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="ebm_pretraining")

    wandb.agent(sweep_id, function=run_experiment, count=5)
    """
    model = GraphTransformer().to('cuda')


    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    trainer = Trainer(model, optimizer, scheduler, device)

    trainer.train(dataloader, n_epochs=10, n_steps=10, step_size=1e-3, verbose=True)
    """
main()