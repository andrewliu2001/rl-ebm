import torch
import os
from torch_geometric.data import Data, Batch
import pickle as pkl

def gen_data(path):
    features_list = []
    edge_list = []
    edge_attr_list = []
    
    for i in range(1, len(os.listdir(f'{path}/features'))+1):
        with open(f'{path}/features/X_{i}.pkl', 'rb') as f:
            features_list += pkl.load(f)

        with open(f'{path}/edge_attrs/edge_attrs_{i}.pkl', 'rb') as f:
            edge_attr_list += pkl.load(f)

        with open(f'{path}/edge_indices/edge_indices_{i}.pkl', 'rb') as f:
            edge_list += pkl.load(f)

    data_list = []
    for i in range(len(features_list)):
        edge_index = torch.tensor(edge_list[i], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list[i], dtype=torch.float)
        feature = torch.tensor(features_list[i], dtype=torch.float)
        data = Data(x=feature, edge_index=edge_index, edge_attr=edge_attr)
        data_list.append(data)

    return data_list

if __name__ == '__main__':
    data_list = gen_data('/home/andrew/rl-ebm/data')
    print(data_list[0].x)
