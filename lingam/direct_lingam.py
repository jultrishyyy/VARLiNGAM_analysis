"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import numpy as np
from sklearn.preprocessing import scale
from sklearn.utils import check_array
import time
from .base import _BaseLiNGAM
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import rbf_kernel 

class DirectLiNGAM(_BaseLiNGAM):
    """Implementation of DirectLiNGAM Algorithm [1]_ [2]_

    References
    ----------
    .. [1] S. Shimizu, T. Inazumi, Y. Sogawa, A. Hyvärinen, Y. Kawahara, T. Washio, P. O. Hoyer and K. Bollen.
       DirectLiNGAM: A direct method for learning a linear non-Gaussian structural equation model.
       Journal of Machine Learning Research, 12(Apr): 1225--1248, 2011.
    .. [2] A. Hyvärinen and S. M. Smith. Pairwise likelihood ratios for estimation of non-Gaussian structural eauation models.
       Journal of Machine Learning Research 14:111-152, 2013.
    """

    def __init__(
        self,
        random_state=None,
        prior_knowledge=None,
        apply_prior_knowledge_softly=False,
        measure="pwling",
    ):
        """Construct a DirectLiNGAM model.

        Parameters
        ----------
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        prior_knowledge : array-like, shape (n_features, n_features), optional (default=None)
            Prior knowledge used for causal discovery, where ``n_features`` is the number of features.

            The elements of prior knowledge matrix are defined as follows [1]_:

            * ``0`` : :math:`x_i` does not have a directed path to :math:`x_j`
            * ``1`` : :math:`x_i` has a directed path to :math:`x_j`
            * ``-1`` : No prior knowledge is available to know if either of the two cases above (0 or 1) is true.
        apply_prior_knowledge_softly : boolean, optional (default=False)
            If True, apply prior knowledge softly.
        measure : {'pwling', 'kernel', 'pwling_fast'}, optional (default='pwling')
            Measure to evaluate independence: 'pwling' [2]_ or 'kernel' [1]_.
            For fast execution with GPU, 'pwling_fast' can be used (culingam is required).
        """
        super().__init__(random_state)
        self._Aknw = prior_knowledge
        self._apply_prior_knowledge_softly = apply_prior_knowledge_softly
    
        self._measure = measure

        if self._Aknw is not None:
            self._Aknw = check_array(self._Aknw)
            self._Aknw = np.where(self._Aknw < 0, np.nan, self._Aknw)

    # def fit(self, X):
    #     """Fit the model to X.

    #     Parameters
    #     ----------
    #     X : array-like, shape (n_samples, n_features)
    #         Training data, where ``n_samples`` is the number of samples
    #         and ``n_features`` is the number of features.

    #     Returns
    #     -------
    #     self : object
    #         Returns the instance itself.
    #     """
    #     # Check parameters
    #     X = check_array(X)
    #     n_features = X.shape[1]

    #     # Check prior knowledge
    #     if self._Aknw is not None:
    #         if (n_features, n_features) != self._Aknw.shape:
    #             raise ValueError(
    #                 "The shape of prior knowledge must be (n_features, n_features)"
    #             )
    #         else:
    #             # Extract all partial orders in prior knowledge matrix
    #             if not self._apply_prior_knowledge_softly:
    #                 self._partial_orders = self._extract_partial_orders(self._Aknw)

    #     # Causal discovery
    #     U = np.arange(n_features)
    #     K = []
    #     X_ = np.copy(X)
    #     if self._measure == "kernel":
    #         X_ = scale(X_)

    #     for _ in range(n_features):
    #         if self._measure == "kernel":
    #             m = self._search_causal_order_kernel(X_, U)
    #         elif self._measure == "pwling_fast":
    #             m = self._search_causal_order_gpu(X_.astype(np.float64), U.astype(np.int32))
    #         else:
    #             m = self._search_causal_order(X_, U)
    #         for i in U:
    #             if i != m:
    #                 X_[:, i] = self._residual(X_[:, i], X_[:, m])
    #         K.append(m)
    #         U = U[U != m]
    #         # Update partial orders
    #         if (self._Aknw is not None) and (not self._apply_prior_knowledge_softly):
    #             self._partial_orders = self._partial_orders[
    #                 self._partial_orders[:, 0] != m
    #             ]

    #     self._causal_order = K
    #     return self._estimate_adjacency_matrix(X, prior_knowledge=self._Aknw)


    def fit(self, X):
        """Fit the model to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        fit_start_time = time.time()
        print("[DirectLiNGAM fit] Starting...")

        # --- Initialization and Checks ---
        init_start_time = time.time()
        # Check parameters
        X = check_array(X)
        n_features = X.shape[1]
        print(f"[DirectLiNGAM fit] Input data shape: {X.shape}")

        # Check prior knowledge
        if self._Aknw is not None:
            print("[DirectLiNGAM fit] Processing prior knowledge...")
            if (n_features, n_features) != self._Aknw.shape:
                raise ValueError(
                    "The shape of prior knowledge must be (n_features, n_features)"
                )
            else:
                # Extract all partial orders in prior knowledge matrix
                if not self._apply_prior_knowledge_softly:
                    self._partial_orders = self._extract_partial_orders(self._Aknw)
                    print("[DirectLiNGAM fit] Extracted partial orders from prior knowledge.")
        init_end_time = time.time()
        print(f"[DirectLiNGAM fit] Initialization and checks took: {init_end_time - init_start_time:.4f} seconds")
        # --- End Initialization ---

        # --- Causal Discovery Loop ---
        causal_discovery_start_time = time.time()
        print("[DirectLiNGAM fit] Starting causal discovery loop...")
        U = np.arange(n_features)
        K = []
        X_ = np.copy(X)
        if self._measure == "kernel":
            print("[DirectLiNGAM fit] Scaling data for kernel measure.")
            X_ = scale(X_) # Scale for kernel method

        total_search_time = 0
        total_residual_time = 0

        for k_idx in range(n_features):
            iter_start_time = time.time()
            print(f"[DirectLiNGAM fit] Causal discovery iteration {k_idx+1}/{n_features}...")

            # --- Search for causal order ---
            search_start_time = time.time()
            if self._measure == "kernel":
                # This step involves independence tests (e.g., mutual information)
                print(f"[DirectLiNGAM fit]   Searching causal order (kernel)... Remaining vars: {len(U)}")
                m = self._search_causal_order_kernel(X_, U)
            elif self._measure == "pwling_fast":
                print(f"[DirectLiNGAM fit]   Searching causal order (pwling_fast/GPU)... Remaining vars: {len(U)}")
                m = self._search_causal_order_gpu(X_.astype(np.float64), U.astype(np.int32))
            elif self._measure == "spearman": 
                print(f"[DirectLiNGAM fit]   Searching causal order (spearman)... Remaining vars: {len(U)}")
                m = self._search_causal_order_spearman(X_, U) 
            elif self._measure == "resid_ng":
                print(f"[DirectLiNGAM fit]   Searching causal order (resid_ng)... Remaining vars: {len(U)}")
                m = self._search_causal_order_resid_ng(X_, U)
            else:
                print(f"[DirectLiNGAM fit]   Searching causal order (default)... Remaining vars: {len(U)}")
                m = self._search_causal_order(X_, U)
            search_end_time = time.time()
            search_duration = search_end_time - search_start_time
            total_search_time += search_duration
            print(f"[DirectLiNGAM fit]   Found variable {m} in causal order. Search took: {search_duration:.4f} seconds.")
            # --- End Search ---

            # --- Calculate residuals ---
            residual_start_time = time.time()
            residual_calcs = 0
            for i in U:
                if i != m:
                    X_[:, i] = self._residual(X_[:, i], X_[:, m])
                    residual_calcs += 1
            residual_end_time = time.time()
            residual_duration = residual_end_time - residual_start_time
            total_residual_time += residual_duration
            print(f"[DirectLiNGAM fit]   Calculated {residual_calcs} residuals. Residual calculation took: {residual_duration:.4f} seconds.")
            # --- End Residual Calculation ---

            K.append(m)
            U = U[U != m]
            # Update partial orders if using prior knowledge
            if (self._Aknw is not None) and (not self._apply_prior_knowledge_softly):
                self._partial_orders = self._partial_orders[self._partial_orders[:, 0] != m]

            iter_end_time = time.time()
            print(f"[DirectLiNGAM fit]   Iteration {k_idx+1} took: {iter_end_time - iter_start_time:.4f} seconds.")

        causal_discovery_end_time = time.time()
        print(f"[DirectLiNGAM fit] Causal discovery loop finished.")
        print(f"[DirectLiNGAM fit]   Total time searching causal order: {total_search_time:.4f} seconds.")
        print(f"[DirectLiNGAM fit]   Total time calculating residuals: {total_residual_time:.4f} seconds.")
        print(f"[DirectLiNGAM fit] Total causal discovery loop time: {causal_discovery_end_time - causal_discovery_start_time:.4f} seconds")
        # --- End Causal Discovery Loop ---

        self._causal_order = K
        print(f"[DirectLiNGAM fit] Determined causal order: {self._causal_order}")

        # --- Estimate Adjacency Matrix ---
        estimate_adj_start_time = time.time()
        print("[DirectLiNGAM fit] Estimating final adjacency matrix...")
        # This step typically involves regression based on the found order
        final_self = self._estimate_adjacency_matrix(X, prior_knowledge=self._Aknw)
        estimate_adj_end_time = time.time()
        print(f"[DirectLiNGAM fit] Adjacency matrix estimation took: {estimate_adj_end_time - estimate_adj_start_time:.4f} seconds")
        # --- End Estimate Adjacency Matrix ---

        fit_end_time = time.time()
        print(f"[DirectLiNGAM fit] Finished. Total time: {fit_end_time - fit_start_time:.4f} seconds")

        return final_self # Return the result of _estimate_adjacency_matrix

    def _extract_partial_orders(self, pk):
        """Extract partial orders from prior knowledge."""
        path_pairs = np.array(np.where(pk == 1)).transpose()
        no_path_pairs = np.array(np.where(pk == 0)).transpose()

        # Check for inconsistencies in pairs with path
        check_pairs = np.concatenate([path_pairs, path_pairs[:, [1, 0]]])
        if len(check_pairs) > 0:
            pairs, counts = np.unique(check_pairs, axis=0, return_counts=True)
            if len(pairs[counts > 1]) > 0:
                raise ValueError(
                    f"The prior knowledge contains inconsistencies at the following indices: {pairs[counts>1].tolist()}"
                )

        # Check for inconsistencies in pairs without path.
        # If there are duplicate pairs without path, they cancel out and are not ordered.
        check_pairs = np.concatenate([no_path_pairs, no_path_pairs[:, [1, 0]]])
        if len(check_pairs) > 0:
            pairs, counts = np.unique(check_pairs, axis=0, return_counts=True)
            check_pairs = np.concatenate([no_path_pairs, pairs[counts > 1]])
            pairs, counts = np.unique(check_pairs, axis=0, return_counts=True)
            no_path_pairs = pairs[counts < 2]

        check_pairs = np.concatenate([path_pairs, no_path_pairs[:, [1, 0]]])
        if len(check_pairs) == 0:
            # If no pairs are extracted from the specified prior knowledge,
            return check_pairs

        pairs = np.unique(check_pairs, axis=0)
        return pairs[:, [1, 0]]  # [to, from] -> [from, to]

    def _residual(self, xi, xj):
        """The residual when xi is regressed on xj."""
        return xi - (np.cov(xi, xj, bias=True)[0, 1] / np.var(xj)) * xj

    def _entropy(self, u):
        """Calculate entropy using the maximum entropy approximations."""
        k1 = 79.047
        k2 = 7.4129
        gamma = 0.37457
        return (1 + np.log(2 * np.pi)) / 2 - k1 * (
            np.mean(np.log(np.cosh(u))) - gamma) ** 2 - k2 * (np.mean(u * np.exp((-(u ** 2)) / 2))) ** 2

    def _diff_mutual_info(self, xi_std, xj_std, ri_j, rj_i):
        """Calculate the difference of the mutual informations."""
        return (self._entropy(xj_std) + self._entropy(ri_j / np.std(ri_j))) - (
            self._entropy(xi_std) + self._entropy(rj_i / np.std(rj_i))
        )

    def _search_candidate(self, U):
        """Search for candidate features"""
        # If no prior knowledge is specified, nothing to do.
        if self._Aknw is None:
            return U, []

        # Apply prior knowledge in a strong way
        if not self._apply_prior_knowledge_softly:
            if len(self._partial_orders) != 0:
                Uc = [i for i in U if i not in self._partial_orders[:, 1]]
                return Uc, []
            else:
                return U, []

        # Find exogenous features
        Uc = []
        for j in U:
            index = U[U != j]
            if self._Aknw[j][index].sum() == 0:
                Uc.append(j)

        # Find endogenous features, and then find candidate features
        if len(Uc) == 0:
            U_end = []
            for j in U:
                index = U[U != j]
                if np.nansum(self._Aknw[j][index]) > 0:
                    U_end.append(j)

            # Find sink features (original)
            for i in U:
                index = U[U != i]
                if self._Aknw[index, i].sum() == 0:
                    U_end.append(i)
            Uc = [i for i in U if i not in set(U_end)]

        # make V^(j)
        Vj = []
        for i in U:
            if i in Uc:
                continue
            if self._Aknw[i][Uc].sum() == 0:
                Vj.append(i)
        return Uc, Vj

    def _search_causal_order(self, X, U):
        """Search the causal ordering."""
        Uc, Vj = self._search_candidate(U)
        if len(Uc) == 1:
            return Uc[0]

        M_list = []
        for i in Uc:
            M = 0
            for j in U:
                if i != j:
                    xi_std = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
                    xj_std = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])
                    ri_j = (
                        xi_std
                        if i in Vj and j in Uc
                        else self._residual(xi_std, xj_std)
                    )
                    rj_i = (
                        xj_std
                        if j in Vj and i in Uc
                        else self._residual(xj_std, xi_std)
                    )
                    M += np.min([0, self._diff_mutual_info(xi_std, xj_std, ri_j, rj_i)]) ** 2
            M_list.append(-1.0 * M)
        return Uc[np.argmax(M_list)]

    def _search_causal_order_gpu(self, X, U):
        """Accelerated Causal ordering.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.
        U: indices of cols in X

        Returns
        -------
        self : object
            Returns the instance itself.
        mlist: causal ordering
        """
        cols = len(U)
        rows = len(X)

        arr = X[:, np.array(U)]
        from lingam_cuda import causal_order as causal_order_gpu
        mlist = causal_order_gpu(arr, rows, cols)
        return U[np.argmax(mlist)]

    def _mutual_information(self, x1, x2, param):
        """Calculate the mutual informations."""
        kappa, sigma = param
        n = len(x1)
        X1 = np.tile(x1, (n, 1))
        K1 = np.exp(-1 / (2 * sigma ** 2) * (X1 ** 2 + X1.T ** 2 - 2 * X1 * X1.T))
        X2 = np.tile(x2, (n, 1))
        K2 = np.exp(-1 / (2 * sigma ** 2) * (X2 ** 2 + X2.T ** 2 - 2 * X2 * X2.T))

        tmp1 = K1 + n * kappa * np.identity(n) / 2
        tmp2 = K2 + n * kappa * np.identity(n) / 2
        K_kappa = np.r_[np.c_[tmp1 @ tmp1, K1 @ K2], np.c_[K2 @ K1, tmp2 @ tmp2]]
        D_kappa = np.r_[
            np.c_[tmp1 @ tmp1, np.zeros([n, n])], np.c_[np.zeros([n, n]), tmp2 @ tmp2]
        ]

        sigma_K = np.linalg.svd(K_kappa, compute_uv=False)
        sigma_D = np.linalg.svd(D_kappa, compute_uv=False)

        return (-1 / 2) * (np.sum(np.log(sigma_K)) - np.sum(np.log(sigma_D)))

    def _search_causal_order_kernel(self, X, U):
        """Search the causal ordering by kernel method."""
        Uc, Vj = self._search_candidate(U)
        if len(Uc) == 1:
            return Uc[0]

        if X.shape[0] > 1000:
            param = [2e-3, 0.5]
        else:
            param = [2e-2, 1.0]

        Tkernels = []
        for j in Uc:
            Tkernel = 0
            for i in U:
                if i != j:
                    ri_j = (
                        X[:, i]
                        if j in Vj and i in Uc
                        else self._residual(X[:, i], X[:, j])
                    )
                    Tkernel += self._mutual_information(X[:, j], ri_j, param)
            Tkernels.append(Tkernel)

        return Uc[np.argmin(Tkernels)]


    def _search_causal_order_spearman(self, X, U):
        """Search the causal ordering using Spearman rank correlation."""
        Uc, Vj = self._search_candidate(U)
        if len(Uc) == 1:
            return Uc[0]

        # Store the sum of squared Spearman correlations for each candidate
        spearman_stat_list = []

        for i in Uc: # Iterate through candidate exogenous variables
            current_spearman_stat = 0
            xi = X[:, i]

            for j in U: # Iterate through all other remaining variables
                if i == j:
                    continue

                xj = X[:, j]

                # Calculate residual ri_j = xi regressed on xj
                # Handle prior knowledge influence on residual calculation if necessary
                # Based on _search_causal_order, this check might be needed:
                if i in Vj and j in Uc:
                    # If j is a candidate exogenous and i is known not to be caused by j,
                    # don't regress, use original xi. But DirectLiNGAM aims for xi indep. of ri_j.
                    # The original DirectLiNGAM logic focuses on finding xi independent of residuals.
                    # So we should probably always calculate the residual here for the test.
                    # Revisit this logic carefully based on the exact interpretation needed.
                    # Let's stick to calculating the residual for the independence test:
                    ri_j = self._residual(xi, xj)
                else:
                    ri_j = self._residual(xi, xj)

                # Calculate Spearman correlation between xi and the residual ri_j
                # Handle cases where residual variance might be zero (constant residual)
                if np.std(ri_j) > 1e-10: # Check for non-zero variance
                    corr, p_value = spearmanr(xi, ri_j)
                    # Check if correlation is NaN (can happen with constant inputs)
                    if not np.isnan(corr):
                        current_spearman_stat += corr**2
                # If residual has zero variance, its correlation is undefined/zero,
                # contributing 0 to the sum, which seems reasonable.

            spearman_stat_list.append(current_spearman_stat)

        # Choose the variable 'i' that minimizes the sum of squared correlations
        # This variable is considered the 'most independent' of its residuals
        # in terms of monotonic correlation.
        min_stat_index = np.argmin(spearman_stat_list)
        return Uc[min_stat_index]
    

    def _center_kernel_matrix(self, K):
        """Centers a kernel matrix K using the centering matrix H."""
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        # K_centered = H @ K @ H # Original formula
        # More efficient calculation:
        mean_k_rows = np.mean(K, axis=1, keepdims=True)
        mean_k_cols = np.mean(K, axis=0, keepdims=True)
        mean_k_all = np.mean(K)
        K_centered = K - mean_k_rows - mean_k_cols + mean_k_all
        return K_centered

    def _calculate_hsic(self, X, Y, kernel='rbf', sigma=1.0):
        """Calculates a basic empirical HSIC statistic.

        Note: This is a simple O(n^2) implementation. For speed, use
            approximations like RFF or libraries like hyppo.
        """
        n_samples = X.shape[0]
        if n_samples < 2:
            return 0.0 # HSIC undefined for < 2 samples

        # Ensure inputs are 2D
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y

        # --- Compute Kernel Matrices ---
        if kernel == 'rbf':
            # gamma = 1 / (2 * sigma**2) # sklearn's gamma definition
            # K = rbf_kernel(X, gamma=gamma)
            # L = rbf_kernel(Y, gamma=gamma)
            # Simplified sigma usage (adjust based on kernel function needs)
            K = rbf_kernel(X, X, gamma=1.0 / (2 * sigma**2)) # Common sigma usage
            L = rbf_kernel(Y, Y, gamma=1.0 / (2 * sigma**2))
        elif kernel == 'linear':
            K = X @ X.T
            L = Y @ Y.T
        else:
            raise ValueError(f"Unsupported kernel for HSIC: {kernel}")

        # --- Center Kernel Matrices ---
        Kc = self._center_kernel_matrix(K)
        Lc = self._center_kernel_matrix(L)

        # --- Calculate HSIC (unbiased estimator variant) ---
        # Formula: HSIC = (1/(n-1)^2) * tr(Kc @ Lc) - slightly different from biased version
        # Simpler biased estimator: HSIC = (1/n^2) * tr(Kc @ Lc)
        # Let's use the simpler biased version here for clarity, scale doesn't matter for argmin
        hsic_value = np.sum(Kc * Lc) # Equivalent to trace(Kc @ Lc.T) for element-wise product

        # Normalize (optional, might help stability but not needed for argmin)
        # hsic_value /= (n_samples**2)

        # Ensure non-negative (due to potential floating point issues)
        return max(0, hsic_value)
    
    def _search_causal_order_hsic(self, X, U):
        """Search the causal ordering using HSIC."""
        Uc, Vj = self._search_candidate(U)
        if len(Uc) == 1:
            return Uc[0]

        # Store the sum of HSIC values for each candidate
        hsic_stat_list = []

        # Get HSIC parameters (example using defaults if not set in __init__)
        hsic_kernel = getattr(self, '_hsic_kernel', 'rbf')
        hsic_sigma = getattr(self, '_hsic_sigma', 1.0) # Adjust default sigma as needed

        for i in Uc: # Iterate through candidate exogenous variables
            current_hsic_stat = 0
            xi = X[:, i]

            for j in U: # Iterate through all other remaining variables
                if i == j:
                    continue

                xj = X[:, j]
                # Calculate residual ri_j = xi regressed on xj
                # Handle prior knowledge (similar logic as in Spearman example)
                ri_j = self._residual(xi, xj)

                # Calculate HSIC between xi and the residual ri_j
                # Handle cases where residual variance might be zero
                if np.std(ri_j) > 1e-10: # Check for non-zero variance
                    hsic_val = self._calculate_hsic(xi, ri_j, kernel=hsic_kernel, sigma=hsic_sigma)
                    current_hsic_stat += hsic_val
                # If residual is constant, HSIC should be ~0, contributing 0.

            hsic_stat_list.append(current_hsic_stat)

        # Choose the variable 'i' that minimizes the sum of HSIC stats
        min_stat_index = np.argmin(hsic_stat_list)
        return Uc[min_stat_index]
    
# --- NEW SKEWNESS METHOD ---
    def _standardize_array(self, arr):
        """Standardizes a 1D array (mean 0, std 1). Returns original if std is near zero."""
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-12:
            return arr - mean # Return mean-centered if std is zero
        return (arr - mean) / std
    
    def _abs_skewness(self, arr_std):
        """Calculates absolute skewness of a pre-standardized 1D array."""
        if arr_std.shape[0] < 3: return 0.0 # Skewness undefined or unstable for very few samples
        # Ensure arr_std is indeed standardized to avoid issues if it's constant coming in
        if np.std(arr_std) < 1e-12: return 0.0
        return np.abs(np.mean(arr_std**3))
        
    # --- NEW RESIDUAL NON-GAUSSIANITY METHOD ---
    def _search_causal_order_resid_ng(self, X_current, U_indices):
        """Search causal ordering by maximizing non-Gaussianity of outgoing residuals."""
        Uc_indices, _ = self._search_candidate(U_indices)
        if not Uc_indices.size: Uc_indices = U_indices # Fallback
        if Uc_indices.size == 0: return None # No variables left to order
        if Uc_indices.size == 1: return Uc_indices[0] # Only one candidate

        candidate_scores = [] # Stores (total_ng_score_for_m, m_idx)

        for m_cand_idx in Uc_indices: # Candidate exogenous variable 'm'
            total_ng_score_for_m = 0.0
            
            xm_std_arr = self._standardize_array(X_current[:, m_cand_idx])
            if np.std(xm_std_arr) < 1e-12: # If candidate 'm' is constant
                candidate_scores.append((-np.inf, m_cand_idx)) # Give a very bad score
                continue

            num_valid_residuals = 0
            for j_other_idx in U_indices:
                if m_cand_idx == j_other_idx:
                    continue

                xj_std_arr = self._standardize_array(X_current[:, j_other_idx])
                if np.std(xj_std_arr) < 1e-12: # If other variable is constant
                    continue
                
                # Calculate R_{j <- m} (residual of Xj_std on Xm_std)
                r_j_from_m_raw = self._residual(xj_std_arr, xm_std_arr)
                r_j_from_m_std = self._standardize_array(r_j_from_m_raw)
                
                if np.std(r_j_from_m_std) < 1e-12: # If residual is constant
                    continue
                
                # Using absolute skewness as the non-Gaussianity measure
                ng_score_of_residual = self._abs_skewness(r_j_from_m_std)
                total_ng_score_for_m += ng_score_of_residual
                num_valid_residuals +=1
            
            if num_valid_residuals > 0 :
                 average_ng_score = total_ng_score_for_m / num_valid_residuals
                 candidate_scores.append((average_ng_score, m_cand_idx))
            else: # No valid residuals found for this candidate (e.g. all other vars were constant)
                 candidate_scores.append((-np.inf, m_cand_idx))


        if not candidate_scores: # Should not happen if Uc_indices was not empty
            if Uc_indices.size > 0 : return Uc_indices[0]
            return None

        # Select the candidate m that MAXIMIZES the total_ng_score_for_m
        best_score, best_m_idx = -np.inf, None
        if candidate_scores: # Ensure candidate_scores is not empty
            # Sort by score descending, then by index ascending as a tie-breaker
            candidate_scores.sort(key=lambda x: (-x[0], x[1]))
            best_m_idx = candidate_scores[0][1]
            
        return best_m_idx