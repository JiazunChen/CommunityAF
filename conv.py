import numpy as np
from scipy import sparse as sp
from graph import Graph

class GraphConv:

    def __init__(self, g: Graph, k: int = 3, alpha: float = 0.85):
        self.graph = g
        self.k = k
        self.alpha = alpha
        
        self.normlized_adj_mat = self._normalize_adj(self.graph.adj_mat).astype(np.float32)


    def __repr__(self):
        return f'Conv_{self.k}_{self.alpha}'

    def __str__(self):
        return self.__repr__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: sp.spmatrix):
        init_val = x
        for _ in range(self.k):
            x = self.alpha * (self.normlized_adj_mat @ x) + (1 - self.alpha) * init_val
        return x

    @staticmethod
    def _normalize_adj(adj: sp.spmatrix) -> sp.spmatrix:
        """Symmetrically normalize adjacency matrix."""
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
