import numpy as np
import scipy.sparse as sp
from bimpcc.utils import generate_2D_gradient_matrices
from bimpcc.utils_reg_sinbeta import build_index_sets, build_jacobian_matrices
from bimpcc.nlp import ObjectiveFn, ConstraintFn, OptimizationProblem
from bimpcc.models.typings import Image
from bimpcc.utils_tvq_sinbeta import diagonal_j_rho, build_nabla_u


def _parse_vars(x: np.ndarray, N: int, M: int):
    return (
        x[:N],
        x[N : N + M],
        x[N + M :],
    )


class TVDenRegObjectiveFn(ObjectiveFn):
    def __init__(
        self,
        true_img: np.ndarray,
        gradient_op: np.ndarray,
        epsilon: float = 1e-4,
        parameter_size: int = 1,
    ):
        self.true_img = true_img.flatten()
        self.K = gradient_op
        self.M, self.N = gradient_op.shape
        self.epsilon = epsilon
        self.parameter_size = parameter_size

    def __call__(self, x: np.ndarray) -> float:
        u, q, alpha = self.parse_vars(x)
        return 0.5 * np.linalg.norm(u - self.true_img) ** 2
        # return 0.5 * np.linalg.norm(u - self.true_img) ** 2 + self.epsilon * np.linalg.norm(alpha) ** 2

    def parse_vars(self, x):
        return _parse_vars(x, self.N, self.M)

    def gradient(self, x: np.ndarray) -> float:
        u, q, alpha = self.parse_vars(x)
        return np.concatenate(
            (u - self.true_img, np.zeros(self.M + self.parameter_size))
        )

    def hessian(self, x: np.ndarray) -> float:
        """
        The Hessian of the objective function.

        Must return a full matrix dont know why exactly.
        """
        d = np.concatenate(
            (
                np.ones(self.N),
                np.zeros(self.M + self.parameter_size),
            )
        )
        return np.diag(d)


class StateConstraintFn(ConstraintFn):
    def __init__(
        self,
        noisy_img: np.ndarray,
        gradient_op: np.ndarray,
        parameter_size: int = 1,
        gamma: int = 100,
        rho: float = 1e-3,
        q_param: float = 0.99,
    ):
        self.noisy_img = noisy_img.flatten()
        self.gradient_op = gradient_op
        self.M, self.N = gradient_op.shape
        self.parameter_size = parameter_size
        self.gamma = gamma
        self.rho = rho
        self.q_param = q_param
        self.delta_gamma = (gamma ** (1 - q_param)) * (q_param**q_param)
        self.Id = sp.eye(self.N).tocoo()
        self.KT = (self.gradient_op.T).tocoo()
        self.K = self.gradient_op.tocoo()

        # ---- 1. Build FIXED sparsity superset (4-hop closure) ----
        KTK = (self.KT @ self.K).tocsr()
        KTK2 = (KTK @ KTK).tocsr()
        W_superset = (KTK2 @ KTK2).tocoo()

        # Add diagonal explicitly
        diag_idx = np.arange(self.N)
        diag = sp.coo_matrix(
            (np.zeros(self.N), (diag_idx, diag_idx)), shape=(self.N, self.N)
        )

        W_superset = (W_superset + diag).tocoo()
        W_superset.sum_duplicates()

        # Store the FIXED pattern indices for W_u
        self.W_row = W_superset.row
        self.W_col = W_superset.col
        self.W_nnz = W_superset.nnz

        # Map (row, col) pairs to a flat index [0...W_nnz-1] for fast filling
        # This allows us to put built values into their correct fixed slot
        self.W_keys = self.W_row * self.N + self.W_col

        # Structure for the 'alpha' column (fixed dense column)
        self.alpha_row = np.arange(self.N)
        self.alpha_col = np.zeros(self.N, dtype=int)

    def __call__(self, x: np.ndarray) -> float:
        u, q, alpha = self.parse_vars(x)
        Da = diagonal_j_rho(
            self.K @ u, self.delta_gamma, self.q_param, self.gamma, self.rho
        )
        return (-1 / self.delta_gamma) * (
            self.KT @ Da @ self.K @ u - alpha * (u - self.noisy_img)
        ) + self.K.T @ q

    def parse_vars(self, x):
        return _parse_vars(x, self.N, self.M)

    def jacobian(self, x: np.ndarray):
        u, q, alpha = self.parse_vars(x)
        beta = float(np.asarray(alpha).squeeze())

        # 1. Compute current Jacobian values (sparse)
        W_computed = build_nabla_u(
            u,
            self.K,
            self.q_param,
            beta,
            self.delta_gamma,
            self.gamma,
            self.rho,
            self.N,
            self.M,
        ).tocoo()
        print(alpha)

        # 2. FILL into the FIXED rigid structure
        # We create a new COO matrix using the PRE-COMPUTED rows/cols
        # and populate it with values where they exist.

        # Map computed (r,c) to sorting keys
        comp_keys = W_computed.row * self.N + W_computed.col

        # Find where computed entries belong in our fixed structure
        # (searchsorted requires sorted keys, but W_computed might not be sorted by duplicate summing)
        # So we sort computed first to be safe
        order = np.argsort(comp_keys)
        comp_keys = comp_keys[order]
        comp_data = W_computed.data[order]

        # Indices in self.W_keys where computed values match
        # This relies on self.W_keys being sorted (it is because we created it from a canonical COO)
        # But wait, COO.row/col aren't strictly 1D sorted. Let's ensure strict sort of fixed keys:
        if not hasattr(self, "_keys_sorted"):
            order_fixed = np.argsort(self.W_keys)
            self.W_keys = self.W_keys[order_fixed]
            self.W_row = self.W_row[order_fixed]
            self.W_col = self.W_col[order_fixed]
            self._keys_sorted = True

        idx = np.searchsorted(self.W_keys, comp_keys)

        # Create the data array of fixed size, filled with zeros
        data_fixed = np.zeros(self.W_nnz, dtype=float)

        # Place found values.
        # Safety check: ensure we only map keys that actually exist in superset
        # (Our KT^4 logic is conservative so this should be 100%, but good for debug)
        valid = (idx < self.W_nnz) & (self.W_keys[idx] == comp_keys)
        data_fixed[idx[valid]] = comp_data[valid]

        # Reconstruct W_u with FIXED structure
        W_u = sp.coo_matrix(
            (data_fixed, (self.W_row, self.W_col)), shape=(self.N, self.N)
        )

        # 3. Alpha part (Fixed structure)
        vect = (1 / self.delta_gamma) * (u - self.noisy_img)
        vect_s = sp.coo_matrix(
            (vect.astype(float), (self.alpha_row, self.alpha_col)), shape=(self.N, 1)
        )

        # 4. Final Stack
        # Ensure we don't accidentally merge and lose zeros.
        # But sp.hstack might reorder.
        # Since we use strict Supersets, let hstack do its work, then sum_duplicates
        # (which preserves explicit zeros usually, but let's see).

        jac = sp.hstack([W_u, self.KT, vect_s], format="coo")
        jac.sum_duplicates()
        return jac


class DualConstraintFn(ConstraintFn):
    def __init__(
        self,
        noisy_img: np.ndarray,
        gradient_op: np.ndarray,
        parameter_size: int = 1,
        gamma: int = 100,
    ):
        self.noisy_img = noisy_img.flatten()
        self.gradient_op = gradient_op
        self.parameter_size = parameter_size
        self.M, self.N = gradient_op.shape
        self.gamma = gamma
        self.Id = sp.eye(self.M).tocoo()
        self.Z_P = sp.coo_matrix((self.M, self.parameter_size))

        # ---- 1. Build FIXED sparsity superset for H_u ----
        # H_u structure is dominated by K.
        # Sometimes H_u terms cancel out or become zero based on active sets.
        # We use pattern(K) as the safe superset.
        self.K = self.gradient_op.tocoo()
        self.K.sum_duplicates()

        # Store FIXED pattern indices
        self.H_row = self.K.row
        self.H_col = self.K.col
        self.H_nnz = self.K.nnz

        # Map (row, col) pairs to flat indices for fast filling
        self.H_keys = self.H_row * self.N + self.H_col

        # To ensure binary search works, we need to sort our fixed keys once
        order_fixed = np.argsort(self.H_keys)
        self.H_keys = self.H_keys[order_fixed]
        self.H_row = self.H_row[order_fixed]
        self.H_col = self.H_col[order_fixed]

    def __call__(self, x: np.ndarray) -> float:
        u, q, alpha = self.parse_vars(x)
        K = self.gradient_op.tocoo()
        Ku = K @ u
        A_gamma, I_gamma, S_gamma, L_1, L_2 = build_index_sets(Ku, self.gamma, self.M)
        return q - (A_gamma @ L_1 + S_gamma @ L_2 + self.gamma * I_gamma) @ Ku

    def parse_vars(self, x):
        return _parse_vars(x, self.N, self.M)

    def jacobian(self, x: np.ndarray):
        u, q, alpha = self.parse_vars(x)

        # 1. Compute current H_u (sparse, might have dropped zeros)
        # Note: build_jacobian_matrices usually returns -(A @ L1_dot_K + ...)
        # Ensure we handle the sign correctly relative to the formula
        H_u = build_jacobian_matrices(
            self.gradient_op, u, q, self.gamma, self.M
        ).tocoo()

        # 2. FILL into the FIXED structure
        comp_keys = H_u.row * self.N + H_u.col
        order = np.argsort(comp_keys)
        comp_keys = comp_keys[order]
        comp_data = H_u.data[order]

        idx = np.searchsorted(self.H_keys, comp_keys)

        # Create fixed data array
        data_fixed = np.zeros(self.H_nnz, dtype=float)

        # Map valid entries
        valid = (idx < self.H_nnz) & (self.H_keys[idx] == comp_keys)
        data_fixed[idx[valid]] = comp_data[valid]

        # Reconstruct with FIXED structure
        H_u_fixed = sp.coo_matrix(
            (data_fixed, (self.H_row, self.H_col)), shape=(self.M, self.N)
        )

        # 3. Final Stack
        # Formula: [-H_u, Id, Z_P]
        # We manually apply the negative sign here to the data
        H_u_fixed.data *= -1.0

        jac = sp.hstack([H_u_fixed, self.Id, self.Z_P], format="coo")
        jac.sum_duplicates()
        return jac


class TVqRegularized:
    def __init__(
        self,
        true_img: Image,
        noisy_img: Image,
        epsilon: float = 1e-4,
        parameter_size: int = 1,
        x0: np.ndarray = None,
        q_param: float = 0.99,
        *args,
        **kwargs,
    ):
        Kx, Ky, self.K = generate_2D_gradient_matrices(true_img.shape[0])
        true_img = true_img.flatten()
        noisy_img = noisy_img.flatten()

        M, N = self.K.shape

        self.objective_func = TVDenRegObjectiveFn(
            true_img, self.K, epsilon=epsilon, parameter_size=parameter_size
        )
        self.eq_constraint_funcs = [
            StateConstraintFn(
                noisy_img, self.K, parameter_size=parameter_size, q_param=q_param
            ),
            DualConstraintFn(
                noisy_img, self.K, parameter_size=parameter_size, gamma=100
            ),
        ]
        self.ineq_constraint_funcs = []

        u_bounds = [(0, None)] * N
        q_bounds = [(None, None)] * M
        alpha_bounds = [(0.0, None)] * (parameter_size)
        self.bounds = u_bounds + q_bounds + alpha_bounds

        if x0 is None:
            self.x0 = np.concatenate(
                [
                    noisy_img,
                    # np.random.randn(N),
                    1e-3 * np.ones(M),
                    100 * np.ones(parameter_size),
                ]
            )
        else:
            self.x0 = x0

    def solve(self, max_iter: int = 3000, tol: float = 1e-4, print_level: int = 5):
        nlp = OptimizationProblem(
            self.objective_func,
            self.eq_constraint_funcs,
            self.ineq_constraint_funcs,
        )
        options = {
            "print_level": print_level,
            "max_iter": max_iter,
            "tol": tol,
            "check_derivatives_for_naninf": "yes",
            "sb": "no",  # quita el banner “silencioso”
            "output_file": "ipopt.log",  # guarda todo el log aquí
        }
        return nlp.solve(
            self.x0, self.bounds, options=options, use_jacobian_sparsity=True
        )
