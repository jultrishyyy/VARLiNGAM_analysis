"""
Python implementation of the TM-modified VARLiNGAM algorithm.
Based on: https://sites.google.com/view/sshimizu06/lingam
"""
import itertools
import warnings
import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array, resample
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy.optimize import linear_sum_assignment

# Placeholder for Tsetlin Machine (replace with actual TM library)
class TsetlinMachine:
    def __init__(self, num_clauses, num_features, num_states, s, T):
        self.num_clauses = num_clauses
        self.num_features = num_features
        self.num_states = num_states
        self.s = s  # Specificity parameter
        self.T = T  # Summation target
        self.clauses = []  # Store learned clauses
    
    def fit(self, X, y):
        # Simulate TM training: learn clauses predicting y from binary X
        # In practice, use TM library (e.g., pyTsetlinMachine)
        # For demo, generate dummy clauses
        self.clauses = [
            {'weight': np.random.rand(), 'literals': np.random.choice([0, 1], self.num_features)}
            for _ in range(self.num_clauses)
        ]
        return self
    
    def get_clauses(self):
        # Return clauses as list of dicts: {'weight': float, 'literals': array}
        return self.clauses

# Assuming these are from the original lingam package
from .base import _BaseLiNGAM
from .bootstrap import BootstrapResult
from .direct_lingam import DirectLiNGAM
from .hsic import hsic_test_gamma
from .utils import predict_adaptive_lasso, find_all_paths, calculate_total_effect

class VARLiNGAM:
    """Implementation of TM-modified VAR-LiNGAM Algorithm
    
    References
    ----------
    .. [1] Aapo Hyvärinen, Kun Zhang, Shohei Shimizu, Patrik O. Hoyer.
       Estimation of a Structural Vector Autoregression Model Using Non-Gaussianity.
       Journal of Machine Learning Research, 11: 1709-1731, 2010.
    .. [2] Ole-Christoffer Granmo. The Tsetlin Machine - A Game Theoretic Bandit Driven Approach
       to Optimal Pattern Recognition with Propositional Logic. 2018.
    """

    def __init__(
        self,
        lags=1,
        criterion='bic',
        prune=False,
        pruning_threshold=0.05,
        ar_coefs=None,
        use_tm=True,
        tm_clauses=100,
        tm_states=100,
        tm_s=3.0,
        tm_T=10,
        bin_threshold=0.5,
        random_state=None
    ):
        """Construct a TM-modified VARLiNGAM model.

        Parameters
        ----------
        lags : int, optional (default=1)
            Number of lags.
        criterion : {‘aic’, ‘fpe’, ‘hqic’, ‘bic’, None}, optional (default='bic')
            Criterion to decide the best lags within ``lags``.
            Searching the best lags is disabled if ``criterion`` is ``None``.
        prune : boolean, optional (default=False)
            Whether to prune the adjacency matrices using threshold-based pruning.
        pruning_threshold : float, optional (default=0.05)
            Threshold for pruning coefficients (used if prune=True).
        ar_coefs : array-like, optional (default=None)
            Coefficients of AR model. Shape: (lags, n_features, n_features).
        use_tm : boolean, optional (default=True)
            Whether to use Tsetlin Machine for instantaneous effects. If False, falls back to DirectLiNGAM.
        tm_clauses : int, optional (default=100)
            Number of TM clauses for learning causal relationships.
        tm_states : int, optional (default=100)
            Number of states per TM automaton.
        tm_s : float, optional (default=3.0)
            TM specificity parameter.
        tm_T : int, optional (default=10)
            TM summation target for voting.
        bin_threshold : float, optional (default=0.5)
            Threshold for binarizing residuals for TM input.
        random_state : int, optional (default=None)
            Seed for random number generator.
        """
        self._lags = lags
        self._criterion = criterion
        self._prune = prune
        self._pruning_threshold = pruning_threshold
        self._ar_coefs = check_array(ar_coefs, allow_nd=True) if ar_coefs is not None else None
        self._use_tm = use_tm
        self._tm_clauses = tm_clauses
        self._tm_states = tm_states
        self._tm_s = tm_s
        self._tm_T = tm_T
        self._bin_threshold = bin_threshold
        self._random_state = random_state
        np.random.seed(random_state)

    def fit(self, X):
        """Fit the TM-modified VARLiNGAM model to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        fit_start_time = time.time()
        print("[TM-VarLiNGAM fit] Starting...")

        self._causal_order = None
        self._adjacency_matrices = None

        X = check_array(X)
        n_samples, n_features = X.shape

        # Step 1: Estimate VAR coefficients and residuals
        var_fit_start = time.time()
        M_taus = self._ar_coefs
        if M_taus is None:
            print("[TM-VarLiNGAM fit] Estimating VAR coefficients...")
            M_taus, lags, residuals = self._estimate_var_coefs(X)
            print(f"[TM-VarLiNGAM fit] VAR estimation complete. Lags={lags}")
        else:
            print("[TM-VarLiNGAM fit] Calculating residuals from provided VAR coefficients...")
            lags = M_taus.shape[0]
            residuals = self._calc_residuals(X, M_taus, lags)
            print("[TM-VarLiNGAM fit] Residual calculation complete.")
        var_fit_end = time.time()
        print(f"[TM-VarLiNGAM fit] VAR estimation/Residual calculation took: {var_fit_end - var_fit_start:.4f} seconds")

        # Step 2: Estimate B_0 (instantaneous effects)
        b0_fit_start = time.time()
        if self._use_tm:
            print("[TM-VarLiNGAM fit] Estimating B_0 using Tsetlin Machine...")
            B0, causal_order = self._estimate_b0_tm(residuals)
        else:
            print("[TM-VarLiNGAM fit] Falling back to DirectLiNGAM for B_0...")
            lingam_model = DirectLiNGAM()
            lingam_model.fit(residuals)
            B0 = lingam_model.adjacency_matrix_
            causal_order = lingam_model.causal_order_
        b0_fit_end = time.time()
        print(f"[TM-VarLiNGAM fit] B_0 estimation took: {b0_fit_end - b0_fit_start:.4f} seconds")

        # Step 3: Calculate B_taus (lagged adjacency matrices)
        calc_b_start = time.time()
        print("[TM-VarLiNGAM fit] Calculating lagged adjacency matrices (B_taus)...")
        B_taus = self._calc_b(X, B0, M_taus)
        calc_b_end = time.time()
        print(f"[TM-VarLiNGAM fit] B_taus calculation took: {calc_b_end - calc_b_start:.4f} seconds")

        # Step 4: Pruning (optional)
        pruning_start = time.time()
        if self._prune:
            print("[TM-VarLiNGAM fit] Pruning adjacency matrices...")
            B_taus = self._pruning(X, B_taus, causal_order)
            pruning_end = time.time()
            print(f"[TM-VarLiNGAM fit] Pruning took: {pruning_end - pruning_start:.4f} seconds")
        else:
            pruning_end = pruning_start
            print("[TM-VarLiNGAM fit] Pruning step skipped.")

        # Store results
        self._ar_coefs = M_taus
        self._lags = lags
        self._residuals = residuals
        self._causal_order = causal_order
        self._adjacency_matrices = B_taus

        fit_end_time = time.time()
        print(f"[TM-VarLiNGAM fit] Finished. Total time: {fit_end_time - fit_start_time:.4f} seconds")

        return self

    def _estimate_b0_tm(self, residuals):
        """Estimate B_0 using Tsetlin Machine.

        Parameters
        ----------
        residuals : array-like, shape (n_samples, n_features)
            Residuals from VAR model.

        Returns
        -------
        B0 : array-like, shape (n_features, n_features)
            Estimated instantaneous adjacency matrix.
        causal_order : list
            Estimated causal order of variables.
        """
        n_samples, n_features = residuals.shape

        # Binarize residuals
        Z = (residuals > self._bin_threshold).astype(int)  # Binary features
        literals = np.hstack([Z, 1 - Z])  # Include negated literals: z_i, ~z_i

        # Initialize B_0
        B0 = np.zeros((n_features, n_features))
        causal_order = list(range(n_features))  # Initial order

        # Train TM for each variable to predict e_j(t)
        tm = TsetlinMachine(
            num_clauses=self._tm_clauses,
            num_features=2 * n_features,
            num_states=self._tm_states,
            s=self._tm_s,
            T=self._tm_T
        )

        for j in range(n_features):
            y = Z[:, j]  # Target: presence of e_j(t)
            tm.fit(literals, y)
            clauses = tm.get_clauses()
            # Map clauses to B_0 coefficients
            for clause in clauses:
                weight = clause['weight']
                for i in range(n_features):
                    if clause['literals'][i]:  # z_i included
                        B0[j, i] += weight
                    elif clause['literals'][i + n_features]:  # ~z_i included
                        B0[j, i] -= weight

        # Ensure acyclicity by permuting B_0 to lower-triangular
        W0 = np.eye(n_features) - B0
        cost = -np.abs(W0)  # Maximize diagonal entries
        row_ind, col_ind = linear_sum_assignment(cost)
        P = np.zeros((n_features, n_features))
        P[row_ind, col_ind] = 1
        B0 = P.T @ B0 @ P
        causal_order = col_ind.tolist()  # Update causal order based on permutation

        return B0, causal_order

    def bootstrap(self, X, n_sampling):
        """Evaluate the statistical reliability of DAG based on bootstrapping.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        n_sampling : int
            Number of bootstrapping samples.

        Returns
        -------
        result : VARBootstrapResult
            Result of bootstrapping.
        """
        X = check_array(X)
        n_samples, n_features = X.shape

        # Store initial settings
        ar_coefs = self._ar_coefs
        lags = self._lags
        criterion = self._criterion
        self._criterion = None

        self.fit(X)
        fitted_ar_coefs = self._ar_coefs

        total_effects = np.zeros([n_sampling, n_features, n_features * (1 + self._lags)])
        adjacency_matrices = []

        for i in range(n_sampling):
            sampled_residuals = resample(self._residuals, n_samples=n_samples)
            resampled_X = np.zeros((n_samples, n_features))
            for j in range(n_samples):
                if j < lags:
                    resampled_X[j, :] = sampled_residuals[j]
                    continue
                ar = np.zeros((1, n_features))
                for t, M in enumerate(fitted_ar_coefs):
                    ar += np.dot(M, resampled_X[j - t - 1, :].T).T
                resampled_X[j, :] = ar + sampled_residuals[j]

            self._ar_coefs = ar_coefs
            self._lags = lags
            self.fit(resampled_X)
            am = np.concatenate([*self._adjacency_matrices], axis=1)
            adjacency_matrices.append(am)

            for c, to in enumerate(reversed(self._causal_order)):
                for from_ in self._causal_order[:n_features - (c + 1)]:
                    total_effects[i, to, from_] = self.estimate_total_effect2(
                        n_features, from_, to
                    )
                for lag in range(self._lags):
                    for from_ in range(n_features):
                        total_effects[i, to, from_ + n_features * (lag + 1)] = \
                            self.estimate_total_effect2(n_features, from_, to, lag + 1)

        self._criterion = criterion
        return VARBootstrapResult(adjacency_matrices, total_effects)

    def estimate_total_effect(self, X, from_index, to_index, from_lag=0):
        """Estimate total effect using causal model."""
        X = check_array(X)
        n_features = X.shape[1]

        if from_lag == 0:
            from_order = self._causal_order.index(from_index)
            to_order = self._causal_order.index(to_index)
            if from_order > to_order:
                warnings.warn(
                    f"The estimated causal effect may be incorrect because "
                    f"the causal order of the destination variable (to_index={to_index}) "
                    f"is earlier than the source variable (from_index={from_index})."
                )

        X_joined = np.zeros((X.shape[0], X.shape[1] * (1 + self._lags + from_lag)))
        for p in range(1 + self._lags + from_lag):
            pos = n_features * p
            X_joined[:, pos:pos + n_features] = np.roll(X[:, 0:n_features], p, axis=0)

        am = np.concatenate([*self._adjacency_matrices], axis=1)
        parents = np.where(np.abs(am[from_index]) > 0)[0]
        from_index = from_index if from_lag == 0 else from_index + (n_features * from_lag)
        parents = parents if from_lag == 0 else parents + (n_features * from_lag)
        predictors = [from_index]
        predictors.extend(parents)

        lr = LinearRegression()
        lr.fit(X_joined[:, predictors], X_joined[:, to_index])
        return lr.coef_[0]

    def estimate_total_effect2(self, n_features, from_index, to_index, from_lag=0):
        """Estimate total effect using causal model."""
        if from_lag == 0:
            from_order = self._causal_order.index(from_index)
            to_order = self._causal_order.index(to_index)
            if from_order > to_order:
                warnings.warn(
                    f"The estimated causal effect may be incorrect because "
                    f"the causal order of the destination variable (to_index={to_index}) "
                    f"is earlier than the source variable (from_index={from_index})."
                )

        am = np.concatenate([*self._adjacency_matrices], axis=1)
        am = np.pad(am, [(0, am.shape[1] - am.shape[0]), (0, 0)])
        from_index = from_index if from_lag == 0 else from_index + (n_features * from_lag)
        effect = calculate_total_effect(am, from_index, to_index)
        return effect

    def get_error_independence_p_values(self):
        """Calculate p-value matrix of independence between error variables."""
        nn = self._residuals
        B0 = self._adjacency_matrices[0]
        E = np.dot(np.eye(B0.shape[0]) - B0, nn.T).T
        n_samples, n_features = E.shape

        p_values = np.zeros([n_features, n_features])
        for i, j in itertools.combinations(range(n_features), 2):
            _, p_value = hsic_test_gamma(
                np.reshape(E[:, i], [n_samples, 1]), np.reshape(E[:, j], [n_samples, 1])
            )
            p_values[i, j] = p_value
            p_values[j, i] = p_value
        return p_values

    def _estimate_var_coefs(self, X):
        """Estimate coefficients of VAR."""
        if self._criterion not in ["aic", "fpe", "hqic", "bic"]:
            var = VAR(X)
            result = var.fit(maxlags=self._lags, trend="n")
        else:
            min_value = float("Inf")
            result = None
            for lag in range(1, self._lags + 1):
                var = VAR(X)
                fitted = var.fit(maxlags=lag, ic=None, trend="n")
                value = getattr(fitted, self._criterion)
                if value < min_value:
                    min_value = value
                    result = fitted
        return result.coefs, result.k_ar, result.resid

    def _calc_residuals(self, X, M_taus, lags):
        """Calculate residuals."""
        X = X.T
        n_features, n_samples = X.shape
        residuals = np.zeros((n_features, n_samples))
        for t in range(n_samples):
            if t - lags < 0:
                continue
            estimated = np.zeros((n_features, 1))
            for tau in range(1, lags + 1):
                estimated += np.dot(M_taus[tau - 1], X[:, t - tau].reshape((-1, 1)))
            residuals[:, t] = X[:, t] - estimated.reshape((-1,))
        residuals = residuals[:, lags:].T
        return residuals

    def _calc_b(self, X, B0, M_taus):
        """Calculate B_taus."""
        n_features = X.shape[1]
        B_taus = np.array([B0])
        for M in M_taus:
            B_t = np.dot((np.eye(n_features) - B0), M)
            B_taus = np.append(B_taus, [B_t], axis=0)
        return B_taus

    def _pruning(self, X, B_taus, causal_order):
        """Prune edges by applying an absolute threshold to B_taus coefficients."""
        threshold = self._pruning_threshold
        print(f"[Pruning] Applying threshold-based pruning with threshold={threshold}.")
        pruned_B_taus = np.copy(B_taus)
        pruned_B_taus[np.abs(pruned_B_taus) < threshold] = 0
        return pruned_B_taus

    @property
    def causal_order_(self):
        """Estimated causal ordering."""
        return self._causal_order

    @property
    def adjacency_matrices_(self):
        """Estimated adjacency matrices."""
        return self._adjacency_matrices

    @property
    def residuals_(self):
        """Residuals of regression."""
        return self._residuals


class VARBootstrapResult(BootstrapResult):
    """Result of bootstrapping for TM-modified VARLiNGAM algorithm."""

    def __init__(self, adjacency_matrices, total_effects):
        """Construct a VARBootstrapResult."""
        super().__init__(adjacency_matrices, total_effects)

    def get_paths(self, from_index, to_index, from_lag=0, to_lag=0, min_causal_effect=None):
        """Get paths and their bootstrap probabilities."""
        if min_causal_effect is None:
            min_causal_effect = 0.0
        else:
            if not 0.0 < min_causal_effect:
                raise ValueError("min_causal_effect must be greater than 0.")
        if to_lag > from_lag:
            raise ValueError("from_lag should be greater than or equal to to_lag.")
        if to_lag == from_lag and to_index == from_index:
            raise ValueError("The same variable is specified for from and to.")

        paths_list = []
        effects_list = []
        for am in self._adjacency_matrices:
            expansion_m = np.zeros((am.shape[1], am.shape[1]))
            n_features = am.shape[0]
            n_lags = int(am.shape[1] / am.shape[0]) - 1
            for i in range(n_lags + 1):
                for j in range(i, n_lags + 1):
                    row = n_features * i
                    col = n_features * j
                    lag = col - row
                    expansion_m[row:row + n_features, col:col + n_features] = \
                        am[0:n_features, lag:lag + n_features]
            paths, effects = find_all_paths(
                expansion_m,
                int(n_features * from_lag + from_index),
                int(n_features * to_lag + to_index),
                min_causal_effect
            )
            paths_list.extend(["_".join(map(str, p)) for p in paths])
            effects_list.extend(effects)

        paths_list = np.array(paths_list)
        effects_list = np.array(effects_list)
        paths_str, counts = np.unique(paths_list, axis=0, return_counts=True)
        order = np.argsort(-counts)
        probs = counts[order] / len(self._adjacency_matrices)
        paths_str = paths_str[order]
        effects = [np.median(effects_list[np.where(paths_list == p)]) for p in paths_str]

        result = {
            "path": [[int(i) for i in p.split("_")] for p in paths_str],
            "effect": effects,
            "probability": probs.tolist()
        }
        return result