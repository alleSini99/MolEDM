"""E(n)-Equivariant Graph Neural Network used as the denoiser.

Satorras et al., "E(n) Equivariant Graph Neural Networks" (2021).  Dense
implementation: molecules are padded to a fixed size and treated as fully
connected graphs, which keeps the code short and is fast enough for QM9
(<= 29 atoms).

Equivariance: messages depend only on squared interatomic distances, and
coordinates are updated along relative-position vectors, so rotating/reflecting
the input rotates/reflects the predicted coordinate noise the same way.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #
def _mlp(sizes, act, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)

class EquivariantConv(nn.Module):
    """One Equivariant Graph Neural Network layer: message passing + a coordinate update."""

    def __init__(self, hidden: int, act=nn.SiLU, coords_range: float = 15.0):
        super().__init__()
        self.coords_range = coords_range

        # Create the MLPs for edge, attention, node, and coordinate updates
        self.edge_mlp = _mlp([2 * hidden + 1, hidden, hidden], act, act())
        self.att_mlp = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.node_mlp = _mlp([2 * hidden, hidden, hidden], act)
        self.coord_mlp = _mlp([hidden, hidden, 1], act)

        # Initializations
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)
        nn.init.zeros_(self.coord_mlp[-1].bias)

    def forward(self, h, x, edge_mask, node_mask):

        # Compute the distances and the norm among the atoms
        diff = x.unsqueeze(2) - x.unsqueeze(1)  #(B, N, N, 3)
        d2 = (diff**2).sum(-1, keepdim=True)  #(B, N, N, 1)
        norm = (d2 + 1e-8).sqrt()  #(B, N, N, 1)

        # Compute the egde features
        n = h.shape[1]
        hi = h.unsqueeze(2).expand(-1, -1, n, -1)  #(B, N, N, H)
        hj = h.unsqueeze(1).expand(-1, n, -1, -1)  #(B, N, N, H)

        # Create the messages
        m = self.edge_mlp(torch.cat([hi, hj, d2], dim=-1))  #(B, N, N, 2*H+1) -> (B, N, N, H)
        m = m * self.att_mlp(m) * edge_mask # (B, N, N, H)

        # Update of the coordinates (equivariant)
        trans = diff / (norm + 1.0) * torch.tanh(self.coord_mlp(m)) * self.coords_range # (B, N, N, 3)
        trans = trans * edge_mask # (B, N, N, 3)
        n_edges = edge_mask.sum(dim=2).clamp(min=1.0) # (B, N, 1)
        x = x + (trans.sum(dim=2) / n_edges) * node_mask # (B, N, 3)

        # Update of the node features (invariant)
        agg = m.sum(dim=2) / n_edges # (B, N, H)
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1)) # (B, N, 2*H) -> (B, N, H)
        return h * node_mask, x * node_mask # (B, N, H), (B, N, 3)


class EGNNDynamics(nn.Module):
    """Predicts the noise (eps_x, eps_h) added to a noisy molecule at time t."""

    def __init__(
        self,
        num_types: int,
        hidden: int = 192,
        n_layers: int = 6,
        act=nn.SiLU,
        coords_range: float = 15.0,
    ):
        super().__init__()

        # Build the model blocks
        self.embed = nn.Linear(num_types + 1, hidden)  # +1 for the timestep
        self.layers = nn.ModuleList(
            EquivariantConv(hidden, act, coords_range / n_layers) for _ in range(n_layers)
        )
        self.decode = _mlp([hidden, hidden, num_types], act)

    def forward(self, t, x, h, node_mask):
        """t: [B] in [0,1]; x: [B,N,3]; h: [B,N,K]; node_mask: [B,N]."""

        # Extract dimensions
        b, n, _ = x.shape
        nm = node_mask.unsqueeze(-1)

        # Craet the edge mask for the fully connected graph (no self-loops)
        eye = torch.eye(n, device=x.device, dtype=x.dtype).view(1, n, n, 1) #(1, N, N, 1)
        edge_mask = (nm.unsqueeze(2) * nm.unsqueeze(1)) * (1.0 - eye)   #(B, N, N, 1)

        # Embed the node features and the timestep
        t_node = t.view(b, 1, 1).expand(-1, n, 1)   #(B, N, 1)
        hidden = self.embed(torch.cat([h, t_node], dim=-1)) * nm   #(B, N, K+1) -> (B, N, H)

        # Run the message passing layers with skipped connections
        x0 = x #(B, N, 3)
        for layer in self.layers:
            hidden, x = layer(hidden, x, edge_mask, nm) # (B, N, H), (B, N, 3)

        # Predict the noise added to the coordinates and atom types
        eps_x = (x - x0) * nm # (B, N, 3)

        # Center the coordinate noise to have zero center of mass, and mask the atom-type noise
        eps_x = eps_x - eps_x.sum(1, keepdim=True) / node_mask.sum(1).view(b, 1, 1) # (B, N, 3)
        eps_x = eps_x * nm # (B, N, 3)
        eps_h = self.decode(hidden) * nm # (B, N, K)
        
        return eps_x, eps_h
