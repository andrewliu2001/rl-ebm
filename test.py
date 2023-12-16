import pickle as pkl

import torch

adjacency_matrix = torch.zeros((5, 5, 3), dtype=torch.float)


def edge_index_from_adjacency(adjacency_matrix):
    """
    Clean an adjacency matrix and convert it to edge_index format.
    """
    adjacency_matrix = adjacency_matrix + adjacency_matrix.transpose(0, 1)
    all_equal = adjacency_matrix[:, :, 0] == adjacency_matrix[:, :, 1]
    for k in range(1, adjacency_matrix.shape[-1]):
        all_equal &= adjacency_matrix[:, :, k-1] == adjacency_matrix[:, :, k]

    # Set the maximum channel to 1 and other channels to 0 unless all channels are equal
    max_indices = torch.argmax(adjacency_matrix, dim=-1)  # Find the indices of the maximum values along the channel dimension
    max_channel_matrix = torch.zeros_like(adjacency_matrix)  # Initialize a matrix of the same shape with zeros

    # Set the maximum channel to 1
    max_channel_matrix.scatter_(2, max_indices.unsqueeze(-1), 1)

    # Keep original values where all channels are equal
    max_channel_matrix[all_equal] = adjacency_matrix[all_equal]

    adjacency_matrix = max_channel_matrix

    num_nodes = adjacency_matrix.size(0)
    
    # Create a grid of coordinates corresponding to the adjacency matrix
    rows, cols = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')
    
    # Mask to find non-zero elements (edges)
    flattened_adjacency_matrix = torch.sum(adjacency_matrix, axis=2)
    mask = flattened_adjacency_matrix != 0
    
    # Get the edge indices
    edge_index = torch.stack([rows[mask], cols[mask]], dim=0)
    
    # Get the corresponding edge attributes
    edge_attr = adjacency_matrix[mask]

    return edge_index, edge_attr

def adjacency_from_edge_index(edge_index, num_nodes, edge_attr):
    # Initialize the adjacency matrix
    adjacency_matrix = torch.zeros((num_nodes, num_nodes, edge_attr.size(1)), dtype=torch.float)

    # Unpack the edge indices
    rows, cols = edge_index

    # Use advanced indexing to assign edge attributes directly
    adjacency_matrix[rows, cols] = edge_attr

    return adjacency_matrix


if __name__ == '__main__':
    
    
    with open('data/edge_indices/edge_indices_1.pkl', 'rb') as f:
        edge_index = torch.from_numpy(pkl.load(f)[0]).long().t().contiguous()
    with open('data/edge_attrs/edge_attrs_1.pkl', 'rb') as f:
        edge_attr = torch.from_numpy(pkl.load(f)[0]).float()
    
    adjacency_matrix = torch.zeros((38, 38, 3), dtype=torch.float)
    adjacency_matrix[edge_index[0], edge_index[1]] = edge_attr.squeeze()

    new_edge_index, new_edge_attr = edge_index_from_adjacency(adjacency_matrix)

    new_adjacency_matrix = adjacency_from_edge_index(new_edge_index, 38, new_edge_attr)

