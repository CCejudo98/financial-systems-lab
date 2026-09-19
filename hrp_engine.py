import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import networkx as nx

class HierarchicalRiskParity:
    """
    Engine de Optimización de Portafolios con HRP, RMT y Teoría de Grafos.
    """
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns.dropna()
        self.cov_matrix = self.returns.cov()
        self.corr_matrix = self.returns.corr()

    def filter_noise_rmt(self) -> pd.DataFrame:
        """
        Limpia la matriz de correlación eliminando ruido estocástico mediante Random Matrix Theory.
        """
        vals, vecs = np.linalg.eigh(self.corr_matrix)
        q = self.returns.shape[0] / self.returns.shape[1]
        sigma = 1.0
        lambda_max = sigma * (1 + (1 / q) + 2 * np.sqrt(1 / q))
        
        vals[vals <= lambda_max] = np.mean(vals[vals <= lambda_max])
        corr_clean = vecs @ np.diag(vals) @ vecs.T
        np.fill_diagonal(corr_clean, 1.0)
        
        return pd.DataFrame(corr_clean, index=self.corr_matrix.index, columns=self.corr_matrix.columns)

    def build_distance_matrix(self, corr: pd.DataFrame) -> pd.DataFrame:
        return np.sqrt(0.5 * (1 - corr))

    def _get_quasi_diag(self, link: np.ndarray) -> list:
        link = link.astype(int)
        sort_ix = [link[-1, 0], link[-1, 1]]
        num_items = link[-1, 3]
        
        while sort_ix[0] >= num_items or sort_ix[1] >= num_items:
            for i, item in enumerate(sort_ix):
                if item >= num_items:
                    sort_ix[i] = link[item - num_items, 0]
                    sort_ix.insert(i + 1, link[item - num_items, 1])
                    break
        return sort_ix

    def _get_cluster_var(self, cov: pd.DataFrame, items: list) -> float:
        cov_slice = cov.iloc[items, items]
        inv_diag = 1.0 / np.diag(cov_slice)
        weights = inv_diag / np.sum(inv_diag)
        return float(weights.T @ cov_slice.values @ weights)

    def allocate(self) -> pd.Series:
        corr_clean = self.filter_noise_rmt()
        dist = self.build_distance_matrix(corr_clean)
        
        dist_compressed = squareform(dist.values)
        link = linkage(dist_compressed, method='single')
        sort_ix = self._get_quasi_diag(link)
        sorted_items = self.corr_matrix.index[sort_ix].tolist()
        
        weights = pd.Series(1.0, index=sorted_items)
        clusters = [sorted_items]
        
        while len(clusters) > 0:
            clusters = [c[j:k] for c in clusters for j, k in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
            for i in range(0, len(clusters), 2):
                c_first = clusters[i]
                c_second = clusters[i + 1]
                
                var_first = self._get_cluster_var(self.cov_matrix, [self.cov_matrix.index.get_loc(x) for x in c_first])
                var_second = self._get_cluster_var(self.cov_matrix, [self.cov_matrix.index.get_loc(x) for x in c_second])
                
                alpha = 1 - var_first / (var_first + var_second)
                weights[c_first] *= alpha
                weights[c_second] *= (1 - alpha)
                
        return weights

    def build_network_graph(self) -> nx.Graph:
        dist = self.build_distance_matrix(self.corr_matrix)
        G = nx.from_pandas_adjacency(dist)
        return nx.minimum_spanning_tree(G)
