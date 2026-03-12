"""
Sampling strategies for latent-space exploration.

Generalized from t2m's TextToMusicSampler. These strategies work on
any latent space — they operate on [N, D] embeddings regardless of
how they were generated (text, image, video conditioning).

Strategies:
    - random: Direct N samples from the flow matching model
    - mean: Single sample closest to distribution centroid
    - peak: Cluster-based selection (kmeans, dbscan, density)
    - diverse: Greedy max-min distance selection
"""

from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class LatentSamplingStrategy:
    """
    Post-hoc selection from a pool of candidate latent embeddings.

    Given N candidate embeddings from the flow matching model, selects
    a representative subset using various strategies.

    Usage:
        strategy = LatentSamplingStrategy(device="cuda")
        selected = strategy.select(candidates, method="dbscan")
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)

    def select(
        self,
        candidates: Tensor,
        method: Literal["mean", "diverse", "kmeans", "dbscan", "density"] = "dbscan",
        max_samples: Optional[int] = None,
    ) -> Tensor:
        """
        Select representative embeddings from candidates.

        Args:
            candidates: [N, D] candidate embeddings.
            method: Selection strategy.
            max_samples: Maximum number of samples to return.

        Returns:
            [K, D] selected embeddings.
        """
        candidates = F.normalize(candidates, dim=-1)
        N = candidates.shape[0]

        if method == "mean":
            return self._select_mean(candidates)
        elif method == "diverse":
            k = max_samples or min(8, N)
            return self._select_diverse(candidates, k)
        elif method == "kmeans":
            return self._select_kmeans(candidates, max_samples)
        elif method == "dbscan":
            return self._select_dbscan(candidates, max_samples)
        elif method == "density":
            return self._select_density(candidates, max_samples)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _select_mean(self, candidates: Tensor) -> Tensor:
        mean = F.normalize(candidates.mean(dim=0), dim=-1)
        idx = (candidates @ mean).argmax().item()
        return candidates[idx : idx + 1]

    def _select_diverse(self, candidates: Tensor, k: int) -> Tensor:
        """Greedy max-min distance selection."""
        N = candidates.shape[0]
        k = min(k, N)

        selected = [np.random.randint(N)]
        for _ in range(k - 1):
            remaining = [i for i in range(N) if i not in selected]
            if not remaining:
                break
            rem = candidates[remaining]
            sel = candidates[selected]
            sims = rem @ sel.T  # [remaining, selected]
            min_sims = sims.min(dim=1)[0]
            best = min_sims.argmin().item()
            selected.append(remaining[best])

        return candidates[selected]

    def _select_kmeans(self, candidates: Tensor, max_samples: Optional[int]) -> Tensor:
        from sklearn.cluster import KMeans

        X = candidates.cpu().numpy()
        N = len(X)
        max_k = min(20, N // 10, N - 1)
        if max_k < 2:
            return candidates

        # Elbow method
        inertias = []
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=5)
            km.fit(X)
            inertias.append(km.inertia_)

        drops = [inertias[i - 1] - inertias[i] for i in range(1, len(inertias))]
        rel_drops = [d / inertias[i] if inertias[i] > 0 else 0 for i, d in enumerate(drops)]
        optimal_k = 2 + (np.argmax(rel_drops) if rel_drops else 0)

        km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        indices = self._cluster_representatives(candidates, labels)
        selected = candidates[indices]
        return selected[:max_samples] if max_samples else selected

    def _select_dbscan(self, candidates: Tensor, max_samples: Optional[int]) -> Tensor:
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors

        X = candidates.cpu().numpy()
        N = len(X)
        k = max(1, min(10, N // 10))

        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        dists, _ = nbrs.kneighbors(X)
        eps = float(np.median(dists[:, -1])) * 1.5

        labels = DBSCAN(eps=eps, min_samples=max(3, N // 50)).fit_predict(X)
        unique = [l for l in np.unique(labels) if l >= 0]

        if not unique:
            return candidates[: max_samples or N]

        if max_samples and len(unique) > max_samples:
            sizes = sorted([(l, np.sum(labels == l)) for l in unique], key=lambda x: -x[1])
            unique = [l for l, _ in sizes[:max_samples]]

        indices = []
        for label in unique:
            mask = labels == label
            cluster = candidates[mask]
            center = F.normalize(cluster.mean(dim=0), dim=-1)
            best = (cluster @ center).argmax().item()
            indices.append(torch.where(torch.from_numpy(mask).to(self.device))[0][best].item())

        return candidates[indices]

    def _select_density(self, candidates: Tensor, max_samples: Optional[int]) -> Tensor:
        from sklearn.neighbors import NearestNeighbors

        X = candidates.cpu().numpy()
        N = len(X)
        k = max(1, min(10, N // 10))

        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
        dists, _ = nbrs.kneighbors(X)
        densities = 1.0 / (dists[:, 1:].mean(axis=1) + 1e-8)

        top_k = max_samples or min(10, N)
        top_indices = np.argsort(densities)[::-1][:top_k]
        return candidates[top_indices]

    def _cluster_representatives(self, candidates: Tensor, labels: np.ndarray) -> List[int]:
        indices = []
        for k in range(labels.max() + 1):
            mask = labels == k
            if not mask.any():
                continue
            cluster = candidates[mask]
            center = F.normalize(cluster.mean(dim=0), dim=-1)
            best = (cluster @ center).argmax().item()
            indices.append(torch.where(torch.from_numpy(mask).to(self.device))[0][best].item())
        return indices
