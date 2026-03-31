from abc import ABC, abstractmethod
from bimpcc.nlp import (
    ConstraintFn,
    ComplementarityConstraintFn,
    ObjectiveFn,
    PenalizedObjectiveFn,
    OptimizationProblem,
)
from typing import List, Tuple, Union
from rich import print
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio as psnr


class MPCCModel(ABC):
    def __init__(
        self,
        objective_func: ObjectiveFn,
        eq_constraint_funcs: List[ConstraintFn],
        ineq_constraint_funcs: List[ConstraintFn],
        complementarity_constraint_func: ComplementarityConstraintFn,
        bounds: List[Tuple[Union[int, None], Union[int, None]]],
        x0: np.ndarray,
        t_init=1.0,
        *args,
        **kwargs,
    ):
        self.parse_vars_fn = objective_func.parse_vars
        self.objective_func = objective_func
        self.eq_constraint_funcs = eq_constraint_funcs
        self.ineq_constraint_funcs = ineq_constraint_funcs
        self.complementarity_constraint_func = complementarity_constraint_func
        self.bounds = bounds
        self.t = t_init
        self.x0 = x0
        self.comp = None

    @abstractmethod
    def compute_complementarity(self, x: np.ndarray) -> float:
        pass

    def _solve_nlp(self, x0, bounds, t, *args, **kwargs):
        print_level = kwargs.get("print_level", 0)
        tol = kwargs.get("tol", 1e-4)
        max_iter = kwargs.get("max_iter", 5000)
        self.complementarity_constraint_func.t = t
        nlp = OptimizationProblem(
            self.objective_func,
            self.eq_constraint_funcs,
            self.ineq_constraint_funcs + [self.complementarity_constraint_func],
        )
        options = {
            "print_level": print_level,
            "tol": tol,
            "max_iter": max_iter,
            "acceptable_tol": 1e-5,
            "constr_viol_tol": 1e-5,
            "nlp_scaling_method": "gradient-based",
            "sb": "no",  # quita el banner “silencioso”
            "output_file": "ipopt_mpcc.log",  # guarda todo el log aquí
        }
        return nlp.solve(x0, bounds, options=options, use_jacobian_sparsity=True)

    def solve(
        self,
        true_img: np.ndarray,
        N: int=128,
        t_values=None,
        tol: float = 1e-3,
        nlp_tol: float = 1e-6,
        nlp_max_iter: int = 5000,
        verbose: bool = False,
        print_level: int = 0,
        *args,
        **kwargs,
    ):


        if t_values is None:
            t_values = [1e-1, 1e-2, 1e-3]

        x = self.x0
        res = None
        fn = None
        history = []

        print(
            f"{'Iter':>5}\t{'Termination_status':>15}\t{'Objective':>15}\t"
            f"{'MPCC_compl':>15}\t{'t':>15}\n"
        )

        for k, t in enumerate(t_values):
            start_time = time.perf_counter()

            res, x_, fn = self._solve_nlp(
                x,
                self.bounds,
                t,
                tol=nlp_tol,
                print_level=print_level,
                max_iter=nlp_max_iter,
            )

            end_time = time.perf_counter()
            time_iter = end_time - start_time

            self.comp = self.compute_complementarity(x_)
            nlp_iter_k = res.get("nit", None)
            alpha_k = float(x_[-1])
            u_k = x_[:N**2]  # ajustar según tu estructura
            u = u_k.reshape((N,N))
            psnr_k = psnr(u, true_img)

            history.append(
                {
                    "k": int(k),
                    "comp": float(self.comp),
                    "nlp_iter": nlp_iter_k,
                    "time": float(time_iter),
                    "obj": float(fn),
                    "t": float(t),
                    "alpha": alpha_k,
                    "psnr": float(psnr_k),
                }
            )

            if verbose:
                print(
                    f"{k:>5}\t{res['status']:>15}\t{fn:>15.6e}\t"
                    f"{self.comp:>15.6e}\t{t:>15.6e}"
                )

            # Si el NLP fue exitoso, usamos la solución como warm start
            if res["status"] >= 0:
                x = x_
            else:
                print(f"Fallo en la iteración {k} para t = {t}")
                res["iter"] = k
                return res, x, fn, history

            # criterio de parada por complementariedad
            if np.abs(self.comp) < tol:
                print(
                    f"{k:>5}*\t{res['status']:>15}\t{fn:>15.6e}\t"
                    f"{self.comp:>15.6e}\t{t:>15.6e}"
                )
                res["iter"] = k
                return res, x_, fn, history

        print(
            f"* (STOPPED) Se resolvieron todos los t_values con complementarity final: "
            f"{self.compute_complementarity(x)}"
        )
        res["iter"] = len(t_values)
        return res, x, fn, history


class MPCCPenalizedModel(ABC):
    def __init__(
        self,
        objective_func: PenalizedObjectiveFn,
        eq_constraint_funcs: List[ConstraintFn],
        ineq_constraint_funcs: List[ConstraintFn],
        bounds: List[Tuple[Union[int, None], Union[int, None]]],
        x0: np.ndarray,
        pi_init: float = 1.0,
        *args,
        **kwargs,
    ):
        self.parse_vars_fn = objective_func.parse_vars
        self.objective_func = objective_func
        self.eq_constraint_funcs = eq_constraint_funcs
        self.ineq_constraint_funcs = ineq_constraint_funcs
        self.bounds = bounds
        self.pi = pi_init
        self.x0 = x0

    @abstractmethod
    def compute_complementarity(self, x: np.ndarray) -> float:
        pass

    def _solve_nlp(self, x0, bounds, pi, mu, tol_c, tol_p, *args, **kwargs):
        print_level = kwargs.get("print_level", 0)
        max_iter = kwargs.get("max_iter", 5000)
        self.objective_func.pi = pi
        nlp = OptimizationProblem(
            self.objective_func,
            self.eq_constraint_funcs,
            self.ineq_constraint_funcs,
        )
        options = {
            "mu_init": mu,
            "mu_strategy": "monotone",
            "dual_inf_tol": tol_p,
            "constr_viol_tol": tol_p,
            "compl_inf_tol": tol_p,
            "print_level": print_level,
            "max_iter": max_iter,
            "acceptable_tol": tol_p,
            "tol": tol_p,
        }
        return nlp.solve(x0, bounds, options=options)

    def solve(
        self,
        max_iter: int = 10,
        tol: float = 1e-8,
        verbose: bool = False,
        print_level: int = 0,
        mu_init: float = 0.1,
        sigma=10,
        gamma=0.4,
        kappa=0.2,
        nu=10,
    ):
        x = self.x0
        pi = self.pi
        mu = mu_init
        info = None
        fn = None

        print(
            f"{'Iter': >5}\t{'Termination_status': >15}\t{'Objective': >15}\t{
                'MPCC_compl': >15}\t{'lg(mu)': >15}\t{'π': >15}\n"
        )

        for k in range(max_iter):
            tol_c = mu**gamma
            tol_p = nu * mu
            info_, x_, fn_ = self._solve_nlp(
                x, self.bounds, pi, mu, tol_c, tol_p, tol=tol, print_level=print_level
            )
            comp = self.compute_complementarity(x_)
            print(
                f"{k: > 5}\t{info_['status']: > 15}\t{fn_: > 15}\t{comp: > 15}\t{
                    np.log10(mu): > 15}\t{pi: > 15}"
            )
            if (np.abs(comp) < tol) & (info_["status"] == 0):
                print(
                    f"Obtained solution satisfies the complementarity condition at {comp} at {k + 1} iterations"
                )
                return info_, x_, fn_
            if (np.abs(comp) <= tol_c) & (info_["status"] >= 0):
                info = info_
                x = x_
                fn = fn_
                # mu *= kappa
            # else:
            #     if pi < 1e10:
            #         pi *= sigma
            #     else:
            #         print("The problem is unbounded")
            #         break
        return info, x, fn
