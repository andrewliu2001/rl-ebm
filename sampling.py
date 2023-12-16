"""

Implements Langevin sampling and HMC

"""

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sat.models import GraphTransformer
import torch.nn.functional as F
import torch.nn as nn
#from graphormer.model import GraphormerEBM
import math
from GAT_model import GAT


def convert_to_data(adjacency_matrix, features):
    """
    Clean an adjacency matrix and convert it to edge_index format.
    """
    data_list = []
    adjacency_matrix = adjacency_matrix * (adjacency_matrix > 0.5)
    adjacency_matrix = adjacency_matrix + adjacency_matrix.transpose(1, 2)
    all_equal = adjacency_matrix[:,:,:,0] == adjacency_matrix[:,:,:,1]
    for k in range(1, adjacency_matrix.shape[-1]):
        all_equal &= adjacency_matrix[:,:,:,k-1] == adjacency_matrix[:,:,:,k]
    max_indices = torch.argmax(adjacency_matrix, dim=-1).to('cuda')
    max_channel_matrix = torch.zeros_like(adjacency_matrix).to('cuda')
    max_channel_matrix.scatter_(3, max_indices.unsqueeze(-1), 1)
    max_channel_matrix[all_equal] = adjacency_matrix[all_equal]
    #print(max_channel_matrix.size())
    #get edge indices
    data_list = []
    for i in range(max_channel_matrix.size(0)):
        adj_i = max_channel_matrix[i]
        edge_index = torch.sum(adj_i,axis=-1).nonzero().t().contiguous().to('cuda')
        #edge_attr = torch.zeros((max_channel_matrix.size(1), max_channel_matrix.size(2), 1)).to('cuda')
        #print(adj_i)
        mask = (adj_i != 0).any(dim=-1)
        edge_attr = adj_i[mask]
        #print(edge_attr.size())
        data_list.append(Data(x=features[i], edge_index=edge_index, edge_attr=edge_attr))
    
    
    return data_list



def langevin_sampling(model, initial_state_X, step_size, n_steps):
    """
    One step of Langevin sampling.
    """
    x = initial_state_X.clone().detach().requires_grad_(True)
    
    for _ in range(n_steps):
        #data_list = convert_to_data(A, x)
        #dataloader = DataLoader(data_list, batch_size=len(data_list), shuffle=False)
        """
        for batch in dataloader:
            batch = batch.to('cuda')
            energy = model(batch)
            #print(energy)
        """
        energy = model(x, A)

        x_grad = torch.autograd.grad(energy, x, grad_outputs=torch.ones_like(energy))[0]
        A_grad = torch.autograd.grad(energy, A, grad_outputs=torch.ones_like(energy), allow_unused=True, create_graph=True)[0]

        with torch.no_grad():
            # Update x by taking a step against the energy gradient
            x -= step_size /2 * x_grad
            #print(x.grad, A.grad)
            A -= step_size /2 * A_grad
            

            # Add isotropic Gaussian noise
            x += 0.005 * torch.randn_like(x)
            A += 0.005 * torch.randn_like(A)

            # Clear the gradients for the next iteration
            x.grad.zero_()
            A.grad.zero_()

    return x.detach(), A.detach()

def sample_ebm(model, init_A, init_X, max_iterations=30, initial_temp=10.0, final_temp=0.001, step_size=100):
    
    state_A = init_A
    state_X = init_X.clone().detach().requires_grad_(True)
    data_list = convert_to_data(state_A, state_X)
    dataloader = DataLoader(data_list, batch_size=len(data_list), shuffle=False)
    for batch in dataloader:
        batch = batch.to('cuda')
        current_energy = model(batch)


    temp = initial_temp

    X_steps = []
    A_steps = []

    for i in range(max_iterations):
        # Cooling schedule: Exponential decay
        temp = initial_temp * (final_temp / initial_temp) ** (i / max_iterations)

        # Take a step of Langevin dynamics on X
        state_X_grad = torch.autograd.grad(current_energy, state_X, grad_outputs=torch.ones_like(current_energy).unsqueeze(0), is_grads_batched=True, retain_graph=True)[0].squeeze(0)

        with torch.no_grad():
                # Update x by taking a step against the energy gradient
                state_X -= step_size /2 * state_X_grad
                
                # Add isotropic Gaussian noise
                state_X += 0.005 * torch.randn_like(state_X)

                # Clear the gradients for the next iteration
                #state_X.grad.zero_()

        
        # Make a random move
        new_state_X = state_X
        new_state_A = state_A + torch.randn_like(state_A)*0.005
        new_data_list = convert_to_data(new_state_A, new_state_X)
        new_dataloader = DataLoader(new_data_list, batch_size=len(new_data_list), shuffle=False)
        for batch in dataloader:
            batch = batch.to('cuda')
            new_energy = model(batch)
            
        # Energy difference
        energy_diff = new_energy - current_energy

        # Probabilistically accept new state
        for i, ed in enumerate(energy_diff):
            if ed.item() < 0 or torch.rand(1).item() < math.exp(-ed.item() / temp):
                state_A[i], current_energy[i] = new_state_A[i], new_energy[i]
        # Stopping criteria could be added here (e.g., if changes are too small)
        #print(current_energy.mean())
        X_steps.append(state_X.detach())
        A_steps.append(state_A.detach())
    return state_A.detach(), state_X.detach(), X_steps, A_steps







if __name__ == '__main__':
    model = GraphTransformer().to('cuda')
    init_A = torch.rand((16, 38, 38, 3)).to('cuda')
    init_X = torch.rand((16, 38, 9)).to('cuda')
    A, X = sample_ebm(model, init_A, init_X)