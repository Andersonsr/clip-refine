#!/usr/bin/env python3
r"""
Estimate Calibrated Temperature T_2 via Monte Carlo Sampling.

This script takes as input:
  - LHS pre-training statistics: positive (mu_1p, sigma_1p) and negative (mu_1n, sigma_1n)
  - RHS downstream statistics:  positive (mu_2p, sigma_2p) and negative (mu_2n, sigma_2n)
  - n1: Pre-training candidate / batch size (e.g., 32768)
  - T1: Pre-training inverse temperature (e.g., 100.0)
  - n2: Target downstream candidate / batch size / class count (e.g., 10, 37, 100, 1000)

It estimates the calibrated temperature T_2 using Monte Carlo sampling via:
  1. Original Softmax formulation (Eq. 8 / Eq. 10): E[p_2(T_2)] = E[p_1(T_1)]
  2. Normalized Log-Softmax formulation: eta_2(T_2) = eta_1(T_1) (Relative Entropy Reduction)
  3. Absolute Log-Softmax formulation: E[ln p_2(T_2)] = E[ln p_1(T_1)] (Expected Loss matching)
  4. Closed-form Quadratic Analytical Approximation (for comparison)

Usage Example:
--------------
python estimate_t2_monte_carlo.py \
    --n1 32768 --t1 100.0 --n2 10 \
    --lhs_mu_p 0.2829 --lhs_sigma_p 0.0468 \
    --lhs_mu_n 0.0825 --lhs_sigma_n 0.0543 \
    --rhs_mu_p 0.2829 --rhs_sigma_p 0.0468 \
    --rhs_mu_n 0.0825 --rhs_sigma_n 0.0543
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import bisect


def get_gauss_hermite_standard_normal(Q: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Compute nodes and normalized weights for standard normal N(0, 1)."""
    y_gh, w_gh = np.polynomial.hermite.hermgauss(Q)
    x_q = np.sqrt(2.0) * y_gh
    omega_q = w_gh / np.sqrt(np.pi)
    return x_q, omega_q


def evaluate_mc_expectations(
    n: int,
    T: float,
    mu_p: float,
    sigma_p: float,
    mu_n: float,
    sigma_n: float,
    K: int = 3000,
    Q: int = 200,
    seed: int = 42,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute Monte Carlo expectations of:
      - Softmax score: p_k = 1 / (1 + exp(ln W - T z_k))
      - Log-Softmax score: ln p_k = -softplus(ln W - T z_k)
      - Loss: -ln p_k
      - Normalized uncertainty reduction: eta = 1 - Loss / ln(n)
    """
    if n <= 1:
        raise ValueError(f"n must be >= 2, got {n}")

    if T <= 0.0:
        ln_n = float(np.log(n))
        return {
            "softmax": 1.0 / n,
            "softmax_se": 0.0,
            "log_softmax": -ln_n,
            "log_softmax_se": 0.0,
            "loss": ln_n,
            "eta_normalized": 0.0,
        }

    x_q, omega_q = get_gauss_hermite_standard_normal(Q)
    x_q_t = torch.tensor(x_q, dtype=torch.float64, device=device)
    omega_q_t = torch.tensor(omega_q, dtype=torch.float64, device=device)
    z_k_q = (mu_p + sigma_p * x_q_t).unsqueeze(0)  # (1, Q)

    gen = torch.Generator(device=device).manual_seed(seed)
    chunk_size = 500

    softmax_chunks = []
    log_softmax_chunks = []

    with torch.no_grad():
        for start in range(0, K, chunk_size):
            cur_k = min(chunk_size, K - start)
            eps = torch.randn((cur_k, n - 1), generator=gen, dtype=torch.float64, device=device)
            z_j = mu_n + sigma_n * eps
            ln_w = torch.logsumexp(T * z_j, dim=1, keepdim=True)  # (cur_k, 1)
            arg = ln_w - T * z_k_q  # (cur_k, Q)

            # Softmax = sigmoid(-arg)
            p_q = torch.sigmoid(-arg)
            softmax_chunks.append(torch.matmul(p_q, omega_q_t))

            # Log-Softmax = -softplus(arg)
            lp_q = -F.softplus(arg)
            log_softmax_chunks.append(torch.matmul(lp_q, omega_q_t))

    s_all = torch.cat(softmax_chunks, dim=0)
    ls_all = torch.cat(log_softmax_chunks, dim=0)

    s_mean = float(torch.mean(s_all).item())
    s_se = float((torch.std(s_all, unbiased=True) / np.sqrt(K)).item())

    ls_mean = float(torch.mean(ls_all).item())
    ls_se = float((torch.std(ls_all, unbiased=True) / np.sqrt(K)).item())

    ln_n = float(np.log(n))
    loss_mean = -ls_mean
    eta = 1.0 - (loss_mean / ln_n) if ln_n > 0 else 0.0

    return {
        "softmax": s_mean,
        "softmax_se": s_se,
        "log_softmax": ls_mean,
        "log_softmax_se": ls_se,
        "loss": loss_mean,
        "eta_normalized": eta,
    }


def solve_root_bisection(
    eval_fn,
    target_value: float,
    t_min: float = 0.01,
    t_max: float = 120.0,
    n_grid: int = 120,
    xtol: float = 1e-5,
) -> Optional[float]:
    """Bracket and solve for T where eval_fn(T) == target_value."""
    t_grid = np.linspace(t_min, t_max, n_grid)
    vals = [eval_fn(t) for t in t_grid]

    bracket = None
    for i in range(len(t_grid) - 1):
        f1 = vals[i] - target_value
        f2 = vals[i + 1] - target_value
        if f1 * f2 <= 0.0:
            bracket = (float(t_grid[i]), float(t_grid[i + 1]))
            break

    if bracket is None:
        return None

    try:
        root = bisect(lambda t: eval_fn(t) - target_value, bracket[0], bracket[1], xtol=xtol, maxiter=80)
        return float(root)
    except Exception:
        return None


def compute_quadratic_t2(
    n1: int,
    T1: float,
    n2: int,
    mu_1p: float,
    sigma_1n: float,
    mu_1n: float,
    mu_2p: float,
    sigma_2n: float,
    mu_2n: float,
) -> Optional[float]:
    """Analytical closed-form quadratic approximation for comparison."""
    delta1 = mu_1p - mu_1n
    delta2 = mu_2p - mu_2n
    log_ratio = math.log((n2 - 1.0) / (n1 - 1.0))

    a = 0.5 * (sigma_2n ** 2)
    b = -delta2
    c = delta1 * T1 - 0.5 * (sigma_1n ** 2) * (T1 ** 2) + log_ratio

    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return None
    r1 = (-b - math.sqrt(disc)) / (2 * a)
    return r1 if r1 > 0 else None


def estimate_t2(
    n1: int,
    T1: float,
    n2: int,
    lhs_mu_p: float,
    lhs_sigma_p: float,
    lhs_mu_n: float,
    lhs_sigma_n: float,
    rhs_mu_p: Optional[float] = None,
    rhs_sigma_p: Optional[float] = None,
    rhs_mu_n: Optional[float] = None,
    rhs_sigma_n: Optional[float] = None,
    K: int = 3000,
    Q: int = 200,
    seed: int = 42,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    Main entry point: Estimates T_2 using Monte Carlo sampling.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Default RHS to LHS if not provided
    if rhs_mu_p is None:
        rhs_mu_p = lhs_mu_p
    if rhs_sigma_p is None:
        rhs_sigma_p = lhs_sigma_p
    if rhs_mu_n is None:
        rhs_mu_n = lhs_mu_n
    if rhs_sigma_n is None:
        rhs_sigma_n = lhs_sigma_n

    delta_1 = lhs_mu_p - lhs_mu_n
    delta_2 = rhs_mu_p - rhs_mu_n

    if verbose:
        print("=" * 78)
        print("MONTE CARLO TEMPERATURE CALIBRATION (T_2 ESTIMATOR)")
        print("=" * 78)
        print("LHS (Pre-training):")
        print(f"  n1 = {n1:<6d} | T1 = {T1:<6.2f}")
        print(f"  Pos: N({lhs_mu_p:.4f}, {lhs_sigma_p:.4f}^2) | Neg: N({lhs_mu_n:.4f}, {lhs_sigma_n:.4f}^2) | Margin Delta_1 = {delta_1:.4f}")
        print("RHS (Downstream Target):")
        print(f"  n2 = {n2:<6d}")
        print(f"  Pos: N({rhs_mu_p:.4f}, {rhs_sigma_p:.4f}^2) | Neg: N({rhs_mu_n:.4f}, {rhs_sigma_n:.4f}^2) | Margin Delta_2 = {delta_2:.4f}")
        print(f"Simulation: K = {K} MC draws, Q = {Q} GH nodes, device = {device}")
        print("-" * 78)

    # 1. Evaluate LHS Targets via Monte Carlo + Gauss-Hermite
    lhs_res = evaluate_mc_expectations(
        n=n1,
        T=T1,
        mu_p=lhs_mu_p,
        sigma_p=lhs_sigma_p,
        mu_n=lhs_mu_n,
        sigma_n=lhs_sigma_n,
        K=K,
        Q=Q,
        seed=seed,
        device=device,
    )

    rho1_target = lhs_res["softmax"]
    psi1_target = lhs_res["log_softmax"]
    loss1_target = lhs_res["loss"]
    eta1_target = lhs_res["eta_normalized"]
    chance_loss_1 = math.log(n1)
    chance_loss_2 = math.log(n2)
    threshold_n2 = math.exp(loss1_target)

    if verbose:
        print("LHS Targets at (n1, T1):")
        print(f"  Expected Softmax Score       E[p_1]   : {rho1_target:.6f} +/- {lhs_res['softmax_se']:.6f}")
        print(f"  Expected Log-Softmax Score   E[ln p_1]: {psi1_target:.6f} +/- {lhs_res['log_softmax_se']:.6f}")
        print(f"  Expected InfoNCE Loss        E[L_1]   : {loss1_target:.6f}")
        print(f"  Normalized Uncertainty Gain  eta_1    : {eta1_target:.6f} ({eta1_target * 100:.2f}% entropy reduction)")
        print(f"  Uniform Chance Loss ln(n1)            : {chance_loss_1:.4f}")
        print(f"  Downstream Chance Loss ln(n2)         : {chance_loss_2:.4f}")
        print("-" * 78)

    # 2. Pre-generate frozen standard normal samples for RHS Sample Average Approximation (SAA)
    gen_rhs = torch.Generator(device=device).manual_seed(seed + 100)
    eps_rhs = torch.randn((K, n2 - 1), generator=gen_rhs, dtype=torch.float64, device=device)
    z_2j = rhs_mu_n + rhs_sigma_n * eps_rhs

    x_q, omega_q = get_gauss_hermite_standard_normal(Q)
    x_q_t = torch.tensor(x_q, dtype=torch.float64, device=device)
    omega_q_t = torch.tensor(omega_q, dtype=torch.float64, device=device)
    z_2k_q = (rhs_mu_p + rhs_sigma_p * x_q_t).unsqueeze(0)

    def eval_rhs_mc(t_val: float) -> Tuple[float, float, float]:
        if t_val <= 0.0:
            return 1.0 / n2, -chance_loss_2, 0.0
        ln_w2 = torch.logsumexp(t_val * z_2j, dim=1, keepdim=True)
        arg = ln_w2 - t_val * z_2k_q
        # Softmax
        p_q = torch.sigmoid(-arg)
        s_val = float(torch.mean(torch.matmul(p_q, omega_q_t)).item())
        # Log-Softmax
        lp_q = -F.softplus(arg)
        ls_val = float(torch.mean(torch.matmul(lp_q, omega_q_t)).item())
        # Normalized gain
        norm_gain = 1.0 - (-ls_val / chance_loss_2) if chance_loss_2 > 0 else 0.0
        return s_val, ls_val, norm_gain

    # 3. Solve Roots via SAA Bisection
    # Formulation A: Original Softmax Probability Matching (Eq. 8 / Eq. 10)
    t2_softmax = solve_root_bisection(lambda t: eval_rhs_mc(t)[0], rho1_target, t_min=0.01, t_max=120.0)

    # Formulation B: Normalized Log-Softmax Matching (Uncertainty Reduction Ratio)
    t2_logsoft_norm = solve_root_bisection(lambda t: eval_rhs_mc(t)[2], eta1_target, t_min=0.01, t_max=120.0)

    # Formulation C: Absolute Log-Softmax Matching (E[L2] = E[L1])
    t2_logsoft_abs = solve_root_bisection(lambda t: eval_rhs_mc(t)[1], psi1_target, t_min=0.01, t_max=120.0)

    # Formulation D: Closed-form Quadratic Analytical Approximation
    t2_quadratic = compute_quadratic_t2(
        n1=n1,
        T1=T1,
        n2=n2,
        mu_1p=lhs_mu_p,
        sigma_1n=lhs_sigma_n,
        mu_1n=lhs_mu_n,
        mu_2p=rhs_mu_p,
        sigma_2n=rhs_sigma_n,
        mu_2n=rhs_mu_n,
    )

    if verbose:
        print("ESTIMATED CALIBRATED TEMPERATURES (T_2):")
        print(f"  1. Softmax Formulation (Eq. 8):         T_2 = {t2_softmax:.4f}" if t2_softmax is not None else "  1. Softmax Formulation (Eq. 8):         No Root")
        print(f"  2. Normalized Log-Softmax (eta2=eta1):  T_2 = {t2_logsoft_norm:.4f}" if t2_logsoft_norm is not None else "  2. Normalized Log-Softmax (eta2=eta1):  No Root")
        if t2_logsoft_abs is not None:
            print(f"  3. Absolute Log-Softmax (Loss match):   T_2 = {t2_logsoft_abs:.4f}")
        else:
            print(f"  3. Absolute Log-Softmax (Loss match):   N/A (n2={n2} < exp(L1)={threshold_n2:.1f}; chance loss {chance_loss_2:.2f} < target loss {loss1_target:.2f})")
        print(f"  4. Closed-Form Quadratic Approximation: T_2 = {t2_quadratic:.4f}" if t2_quadratic is not None else "  4. Closed-Form Quadratic Approximation: No Real Root")
        print("=" * 78)

    return {
        "inputs": {
            "n1": n1,
            "T1": T1,
            "n2": n2,
            "lhs": {"mu_p": lhs_mu_p, "sigma_p": lhs_sigma_p, "mu_n": lhs_mu_n, "sigma_n": lhs_sigma_n},
            "rhs": {"mu_p": rhs_mu_p, "sigma_p": rhs_sigma_p, "mu_n": rhs_mu_n, "sigma_n": rhs_sigma_n},
        },
        "lhs_targets": {
            "softmax_expected": rho1_target,
            "softmax_se": lhs_res["softmax_se"],
            "log_softmax_expected": psi1_target,
            "log_softmax_se": lhs_res["log_softmax_se"],
            "loss_expected": loss1_target,
            "eta_normalized_gain": eta1_target,
            "chance_loss_1": chance_loss_1,
            "chance_loss_2": chance_loss_2,
            "threshold_n2_abs_match": threshold_n2,
        },
        "T2_estimates": {
            "T2_softmax_eq8": t2_softmax,
            "T2_log_softmax_normalized": t2_logsoft_norm,
            "T2_log_softmax_absolute": t2_logsoft_abs,
            "T2_quadratic_analytical": t2_quadratic,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate Calibrated Temperature T_2 via Monte Carlo Sampling from Logit Statistics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Primary Inputs
    parser.add_argument("--n1", type=int, default=32768, help="Pre-training candidate / batch size")
    parser.add_argument("--t1", type=float, default=100.0, help="Pre-training inverse temperature (T1)")
    parser.add_argument("--n2", type=int, required=True, help="Downstream candidate size / batch size / class count")

    # LHS Statistics (Defaults to OpenCLIP ViT-B/16 LAION-2B statistics)
    parser.add_argument("--lhs_mu_p", type=float, default=0.2829, help="LHS positive similarity location (median)")
    parser.add_argument("--lhs_sigma_p", type=float, default=0.0468, help="LHS positive similarity scale (sigma_MAD)")
    parser.add_argument("--lhs_mu_n", type=float, default=0.0825, help="LHS negative similarity location (median)")
    parser.add_argument("--lhs_sigma_n", type=float, default=0.0543, help="LHS negative similarity scale (sigma_MAD)")

    # RHS Statistics (Defaults to LHS if not provided)
    parser.add_argument("--rhs_mu_p", type=float, default=None, help="RHS positive similarity location (median)")
    parser.add_argument("--rhs_sigma_p", type=float, default=None, help="RHS positive similarity scale (sigma_MAD)")
    parser.add_argument("--rhs_mu_n", type=float, default=None, help="RHS negative similarity location (median)")
    parser.add_argument("--rhs_sigma_n", type=float, default=None, help="RHS negative similarity scale (sigma_MAD)")

    # Monte Carlo Hyperparameters
    parser.add_argument("--k_mc", type=int, default=3000, help="Monte Carlo sample size (K)")
    parser.add_argument("--q_gh", type=int, default=200, help="Gauss-Hermite quadrature nodes (Q)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save JSON results")

    args = parser.parse_args()

    results = estimate_t2(
        n1=args.n1,
        T1=args.t1,
        n2=args.n2,
        lhs_mu_p=args.lhs_mu_p,
        lhs_sigma_p=args.lhs_sigma_p,
        lhs_mu_n=args.lhs_mu_n,
        lhs_sigma_n=args.lhs_sigma_n,
        rhs_mu_p=args.rhs_mu_p,
        rhs_sigma_p=args.rhs_sigma_p,
        rhs_mu_n=args.rhs_mu_n,
        rhs_sigma_n=args.rhs_sigma_n,
        K=args.k_mc,
        Q=args.q_gh,
        seed=args.seed,
        device=args.device,
        verbose=True,
    )

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to: {out_p}")


if __name__ == "__main__":
     
    
    results_imagenet = estimate_t2(
        n1=32768,
        T1=100.0,
        n2=1000,              # Target: ImageNet-1K (1,000 classes)
        lhs_mu_p=0.3047,
        lhs_sigma_p=0.0485,
        lhs_mu_n=0.0105,
        lhs_sigma_n=0.0595,
        K=3000,
        Q=200,
        verbose=True,
    )
