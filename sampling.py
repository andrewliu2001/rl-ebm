"""

Implements Langevin sampling and HMC

"""

import torch

def langevin_sampling(model, initial_state, step_size, n_steps):
    x = initial_state.clone().detach().requires_grad_(True)

    for _ in range(n_steps):
        energy = model.energy(x)
        energy.backward()

        with torch.no_grad():
            # Update x by taking a step against the energy gradient
            x -= step_size * x.grad

            # Add isotropic Gaussian noise
            x += torch.sqrt(torch.tensor(2 * step_size)) * torch.randn_like(x)

            # Clear the gradients for the next iteration
            x.grad.zero_()

    return x.detach()


def leapfrog_step(x, p, grad, step_size, n_steps):
    """ Perform leapfrog steps for HMC. """
    x = x.clone()
    p = p.clone()

    p -= step_size * grad(x) / 2  # half step for momentum
    for _ in range(n_steps - 1):
        x += step_size * p  # full step for position
        p -= step_size * grad(x)  # full step for momentum
    x += step_size * p
    p -= step_size * grad(x) / 2  # half step for momentum

    return x, p

def hmc_sampling(energy_function, initial_x, step_size, n_steps, n_samples):
    """ Perform HMC sampling from a given energy-based model. """
    samples = []
    x = initial_x

    for _ in range(n_samples):
        # current state
        current_energy = energy_function(x)
        current_grad = torch.autograd.grad(current_energy, x)[0]

        # proposed state
        p = torch.randn_like(x)  # initial momentum is N(0, 1)
        x_new, p_new = leapfrog_step(x, p, lambda x: torch.autograd.grad(energy_function(x), x)[0], step_size, n_steps)
        
        # Metropolis-Hastings step
        new_energy = energy_function(x_new)
        new_grad = torch.autograd.grad(new_energy, x_new)[0]

        # Acceptance probability
        current_hamiltonian = current_energy + 0.5 * p.pow(2).sum()
        new_hamiltonian = new_energy + 0.5 * p_new.pow(2).sum()
        accept_prob = torch.exp(current_hamiltonian - new_hamiltonian)

        if torch.rand(1) < accept_prob:
            x = x_new  # accept the new state

        samples.append(x.detach())

    return samples