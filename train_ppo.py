from sat.models import GraphTransformer
from evaluate import postprocess_molecule
from reward import evaluate_molecule
import wandb
from sampling import sample_ebm, convert_to_data
import torch
from torch.nn.utils import clip_grad_value_
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PPO_Trainer():
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self, n_epochs, n_steps, step_size, clip_grad, verbose=True):
        """ Train the model. """
        for epoch in range(n_epochs):
            average_reward = self.train_epoch(n_steps, step_size, clip_grad)

            if verbose:
                print(f'Epoch {epoch + 1} | Train reward: {average_reward:.3f}')

            if self.scheduler:
                self.scheduler.step()
              

    def train_epoch(self, n_steps, step_size, clip_grad):
        """ Train the model for one epoch. """
        self.model.train()
    
        init_A = torch.rand(128, 38, 38, 3).to(self.device)
        init_X = torch.rand(128, 38, 9).to(self.device)
        self_A, self_X, X_steps, A_steps = sample_ebm(self.model, init_A, init_X, step_size=step_size, max_iterations=n_steps)

        X_history = torch.stack(X_steps, dim=0).to(self.device)
        A_history = torch.stack(A_steps, dim=0).to(self.device)

        smiles_list, molecule_list = postprocess_molecule(self_X, self_A)
        
        scores = torch.tensor([evaluate_molecule(smile) for smile in smiles_list]).float().reshape(-1, 1).to(self.device)

        
        sum_energy = 0
        
        for i in range(len(X_history)):
          X = X_history[i]
          A = A_history[i]
          data_list = convert_to_data(A, X)
          dataloader = DataLoader(data_list, batch_size=len(data_list), shuffle=False)

          for batch in dataloader:
              batch = batch.to(self.device)
              sum_energy += self.model(batch)

        loss = (sum_energy * scores).mean()
        loss.backward()
        clip_grad_value_(self.model.parameters(), clip_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()
        wandb.log({'Reward': scores.detach().mean().item()})

        return scores.detach().mean().item()

def run_ppo():
    wandb.init(project='ebm_finetuning', entity='andrewliu1324')
    config = wandb.config
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GraphTransformer(num_layers=4, num_heads=8, d_model=64, batch_norm=False).to(device)
    model.load_state_dict(torch.load('checkpoints/model_1.pt'))
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.0, 0.999))
    trainer = PPO_Trainer(model, optimizer, None, device)
    trainer.train(n_epochs=500, n_steps=config.n_steps, step_size=config.step_size, clip_grad=config.clip_grad, verbose=True)

    torch.save(model.state_dict(), f'/home/andrew/rl-ebm/finetuned_checkpoints/finetuned_model.pt')


def main():

    #Set up wandb sweep
    sweep_config = {
        'method': 'bayes',  # Can be grid, random, or bayes
        'metric': {
            'name': 'reward',
            'goal': 'maximize'   
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


    sweep_id = wandb.sweep(sweep_config, project="ebm_finetuning")

    wandb.agent(sweep_id, function=run_ppo, count=1)
    """
    model = GraphTransformer().to('cuda')


    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    trainer = Trainer(model, optimizer, scheduler, device)

    trainer.train(dataloader, n_epochs=10, n_steps=10, step_size=1e-3, verbose=True)
    """
main()