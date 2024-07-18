import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from torch.nn import (
    Module,
    ModuleList,
    Linear,
    Sequential,
)

from functools import partial, wraps

from einops import rearrange, repeat, reduce, einsum, pack, unpack
from einops.layers.torch import Rearrange


class ExpressCoordinatesInFrame(Module):
    """ Algorithm  29 """

    def __init__(
        self,
        eps = 1e-8
    ):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        coords,
        frame
    ):
        """
        coords: coordinates to be expressed in the given frame
        frame: frame defined by three points
        """

        if frame.ndim == 2:
            frame = rearrange(frame, 'fr fc -> 1 1 fr fc')
        elif frame.ndim == 3:
            frame = rearrange(frame, 'b fr fc -> b 1 fr fc')

        # Extract frame atoms
        a, b, c = frame.unbind(dim=-1)
        w1 = F.normalize(a - b, dim=-1, eps=self.eps)
        w2 = F.normalize(c - b, dim=-1, eps=self.eps)

        # Build orthonormal basis
        e1 = F.normalize(w1 + w2, dim=-1, eps=self.eps)
        e2 = F.normalize(w2 - w1, dim=-1, eps=self.eps)
        e3 = torch.cross(e1, e2, dim=-1)

        # Project onto frame basis
        d = coords - b
        transformed_coords = torch.stack(
            [
                einsum(d, e1, '... i, ... i -> ...'),
                einsum(d, e2, '... i, ... i -> ...'),
                einsum(d, e3, '... i, ... i -> ...'),
            ],
            dim=-1,
        )

        return transformed_coords

class ComputeAlignmentError(Module):
    """ Algorithm 30 """

    def __init__(
        self,
        eps: float = 1e-8
    ):
        super().__init__()
        self.eps = eps
        self.express_coordinates_in_frame = ExpressCoordinatesInFrame()

    def forward(
        self,
        pred_coords,
        true_coords,
        pred_frames,
        true_frames
    ):
        """
        pred_coords: predicted coordinates
        true_coords: true coordinates
        pred_frames: predicted frames
        true_frames: true frames
        """
        num_res = pred_coords.shape[1]
        
        pair2seq = partial(rearrange, pattern='b n m ... -> b (n m) ...')
        seq2pair = partial(rearrange, pattern='b (n m) ... -> b n m ...', n = num_res, m = num_res)
        
        pair_pred_coords = pair2seq(repeat(pred_coords, 'b n d -> b n m d', m = num_res))
        pair_true_coords = pair2seq(repeat(true_coords, 'b n d -> b n m d', m = num_res))
        pair_pred_frames = pair2seq(repeat(pred_frames, 'b n d e -> b m n d e', m = num_res))
        pair_true_frames = pair2seq(repeat(true_frames, 'b n d e -> b m n d e', m = num_res))
        
        # Express predicted coordinates in predicted frames
        pred_coords_transformed = self.express_coordinates_in_frame(pair_pred_coords, pair_pred_frames)

        # Express true coordinates in true frames
        true_coords_transformed = self.express_coordinates_in_frame(pair_true_coords, pair_true_frames)

        # Compute alignment errors
        alignment_errors = torch.sqrt(
            torch.sum((pred_coords_transformed - true_coords_transformed) ** 2, dim=-1) + self.eps
        )
        
        alignment_errors = seq2pair(alignment_errors)

        return alignment_errors
    

class construct_ABC(Module):
    def __init__(self):
        super().__init__()

    def forward(self, coords, mask):
        B, N, _ = coords.shape

        # Expand dimensions to calculate pairwise distances
        coords_expanded_1 = coords.unsqueeze(2).expand(B, N, N, 3)
        coords_expanded_2 = coords.unsqueeze(1).expand(B, N, N, 3)

        # Compute pairwise distances
        distances = torch.norm(coords_expanded_1 - coords_expanded_2, dim=3)

        # Since distance to self is zero, we replace it with a large value to exclude it
        distances += torch.eye(N).unsqueeze(0).expand(B, N, N) * 1e9

        # Find the indices of the first and second nearest neighbors
        nearest_neighbors = distances.argsort(dim=2)
        first_nearest_indices = nearest_neighbors[:, :, 0]
        second_nearest_indices = nearest_neighbors[:, :, 1]

        A = torch.gather(coords, 1, first_nearest_indices.unsqueeze(-1).expand(B, N, 3))
        C = torch.gather(coords, 1, second_nearest_indices.unsqueeze(-1).expand(B, N, 3))

        return A, C, first_nearest_indices, second_nearest_indices

if __name__ == '__main__':
    test = construct_ABC()
    
    coords = torch.rand(64, 48, 3)
    A,C, first_idx, second_idx = test(coords=coords,mask=None)

    print(A.shape, C.shape)