import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import networkx as nx

class HierarchicalRiskParity:
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns.dropna()
        self.cov = self.returns.cov()
        self.corr = self.returns.corr()

    def _get_distance_matrix(self) -> np.ndarray:
        # Calcular matriz de distancia angular d_ij = sqrt(2 * (1 - rho_ij))
        dist = np.sqrt(np.clip(2 * (1 - self.corr.values), 0, None))
        # Forzar simetría numérica exacta para evitar errores de precisión flotante en SciPy
        dist = (dist + dist.T) / 2.0
        return dist

    def _get_quasi_diag(self, link: np.ndarray) -> list:
        # Reordenamiento jerárquico de activos
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df1 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df1])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    def _get_cluster_var(self, cov: pd.DataFrame, c_items: list) -> float:
        # Varianza inversa del clúster
        cov_slice = cov.iloc[c_items, c_items]
        ivp = 1.0 / np.diag(cov_slice)
        ivp /= ivp.sum()
        w = ivp.reshape(-1, 1)
        cluster_var = np.dot(np.dot(w.T, cov_slice), w)[0, 0]
        return cluster_var

    def _get_rec_bisection(self, cov: pd.DataFrame, sort_ix: list) -> pd.Series:
        # Bisección recursiva para asignación de pesos
        w = pd.Series(1.0, index=sort_ix)
        c_items = [sort_ix]
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(c_items), 2):
                c_items0 = c_items[i]
                c_items1 = c_items[i + 1]
                var0 = self._get_cluster_var(cov, c_items0)
                var1 = self._get_cluster_var(cov, c_items1)
                alpha = 1.0 - var0 / (var0 + var1)
                w[c_items0] *= alpha
                w[c_items1] *= 1.0 - alpha
        return w

    def allocate(self) -> pd.Series:
        dist = self._get_distance_matrix()
        # Transformar a formato condensado que exige scipy pdist/linkage
        sq_dist = sch.distance.squareform(dist, checks=False)
        link = sch.linkage(sq_dist, method='ward')
        sort_ix = self._get_quasi_diag(link)
        sort_ix = self.corr.index[sort_ix].tolist()
        weights = self._get_rec_bisection(self.cov, sort_ix)
        return weights.sort_index()

    def build_network_graph(self) -> nx.Graph:
        dist = self._get_distance_matrix()
        dist_df = pd.DataFrame(dist, index=self.corr.index, columns=self.corr.columns)
        G = nx.from_pandas_adjacency(dist_df)
        mst = nx.minimum_spanning_tree(G)
        return mst
