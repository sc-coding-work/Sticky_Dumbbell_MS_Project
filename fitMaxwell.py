#!/usr/bin/env python3
"""
Fit generalized Maxwell model to rheology data (G', G'') vs frequency,
with optional frequency-range selection.

Usage:
    - Place your data file (default: data.txt) with 3 columns:
          omega   G'   G''
      (whitespace or comma separated)
    - Optionally set omega_min and omega_max to restrict the fit range.
    - Run:  python fit_maxwell.py
"""
#Note the data file is in format omega G'' omega G' where omega is the frequency
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os

# ------------------ User settings ------------------
filename = "ExptDHR2_310.bbx"     # file with omega, G', G''
n_modes = 2               # number of Maxwell modes to fit
omega_min = 0.0          # e.g. 0.1  — set to None to disable lower bound
omega_max = None          # e.g. 1000 — set to None to disable upper bound
maxfev = 20000            # maximum iterations
eps_for_weight = 1e-8     # avoid divide-by-zero in weighting
# ---------------------------------------------------


def read_data(filepath: str):
    """Read omega, G', G'' from text/CSV (three columns)."""
    df = pd.read_csv(filepath, comment="#", header=None,
                     sep=None, engine="python",
                     names=["omega", "Gpp", "omega2", "Gp"])
    df = df.dropna()
    return df["omega"].values, df["Gp"].values, df["Gpp"].values


def generalized_maxwell_response(omega, Gs, taus):
    """
    Maxwell model:
        G'(ω)=Σ Gk (ω τk)^2 / (1+(ω τk)^2)
        G''(ω)=Σ Gk (ω τk)   / (1+(ω τk)^2)
    """
    omega = np.asarray(omega)[:, None]
    Gs = np.asarray(Gs)[None, :]
    taus = np.asarray(taus)[None, :]
    x = omega * taus
    Gp = np.sum(Gs * (x**2) / (1 + x**2), axis=1)
    Gpp = np.sum(Gs * (x) / (1 + x**2), axis=1)
    return Gp, Gpp

def compute_goodness_of_fit(omega, Gp_exp, Gpp_exp, res, n_modes):
    """Compute R^2, log-space R^2, RMSE, reduced chi^2 and AIC."""
    Gs, taus = pack_params(res.x, n_modes)
    Gp_fit, Gpp_fit = generalized_maxwell_response(omega, Gs, taus)

    # Avoid log of zero or negative
    eps = 1e-12
    Gp_exp_safe = np.maximum(Gp_exp, eps)
    Gpp_exp_safe = np.maximum(Gpp_exp, eps)
    Gp_fit_safe = np.maximum(Gp_fit, eps)
    Gpp_fit_safe = np.maximum(Gpp_fit, eps)

    # Residuals (linear space)
    r_Gp = Gp_exp - Gp_fit
    r_Gpp = Gpp_exp - Gpp_fit

    # RMSE
    rmse_Gp = np.sqrt(np.mean(r_Gp**2))
    rmse_Gpp = np.sqrt(np.mean(r_Gpp**2))

    # R² function
    def r2(y, yfit):
        ss_res = np.sum((y - yfit)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - ss_res/ss_tot

    # Linear-space R²
    r2_Gp = r2(Gp_exp, Gp_fit)
    r2_Gpp = r2(Gpp_exp, Gpp_fit)

    # Log-space R²
    r2_log_Gp = r2(np.log10(Gp_exp_safe), np.log10(Gp_fit_safe))
    r2_log_Gpp = r2(np.log10(Gpp_exp_safe), np.log10(Gpp_fit_safe))

    # Combined R² (linear + log)
    y_all = np.concatenate([Gp_exp, Gpp_exp])
    yfit_all = np.concatenate([Gp_fit, Gpp_fit])
    r2_total = r2(y_all, yfit_all)

    ylog_all = np.concatenate([np.log10(Gp_exp_safe), np.log10(Gpp_exp_safe)])
    ylogfit_all = np.concatenate([np.log10(Gp_fit_safe), np.log10(Gpp_fit_safe)])
    r2_log_total = r2(ylog_all, ylogfit_all)

    # Reduced chi-square (weighted residuals already in res.fun)
    dof = max(1, len(res.fun) - len(res.x))
    chi2_red = np.sum(res.fun**2) / dof

    # AIC
    k = len(res.x)
    n = len(y_all)
    rss = np.sum((y_all - yfit_all)**2)
    aic = n * np.log(rss/n) + 2*k

    return {
        "R2_Gp": r2_Gp,
        "R2_Gpp": r2_Gpp,
        "R2_total": r2_total,
        "R2_log_Gp": r2_log_Gp,
        "R2_log_Gpp": r2_log_Gpp,
        "R2_log_total": r2_log_total,
        "RMSE_Gp": rmse_Gp,
        "RMSE_Gpp": rmse_Gpp,
        "Reduced_chi2": chi2_red,
        "AIC": aic
    }


def pack_params(vector, n):
    """Unpack parameter vector: [G1..Gn, log_tau1..log_taun]."""
    Gs = vector[:n]
    taus = np.exp(vector[n:2 * n])
    return Gs, taus


def residuals_fn(vector, omega, Gp_exp, Gpp_exp, n):
    """Residuals weighted by experimental values."""
    Gs, taus = pack_params(vector, n)
    Gp_mod, Gpp_mod = generalized_maxwell_response(omega, Gs, taus)
    w_Gp = 1.0 / np.maximum(np.abs(Gp_exp), eps_for_weight)
    w_Gpp = 1.0 / np.maximum(np.abs(Gpp_exp), eps_for_weight)
    res1 = (Gp_mod - Gp_exp) * w_Gp
    res2 = (Gpp_mod - Gpp_exp) * w_Gpp
    return np.concatenate([res1, res2])


def initial_guess(omega, Gp, Gpp, n):
    total_G = np.max(np.concatenate([Gp, Gpp]))
    Gs0 = np.full(n, total_G / n)
    taus0 = np.exp(np.linspace(np.log(1 / np.max(omega)),
                               np.log(1 / np.min(omega)), n))
    return np.concatenate([Gs0, np.log(taus0)])


def fit_generalized_maxwell(omega, Gp, Gpp, n_modes=2):
    """Fit Maxwell model by weighted least squares."""
    x0 = initial_guess(omega, Gp, Gpp, n_modes)
    lower = np.concatenate([np.zeros(n_modes), np.full(n_modes, np.log(1e-10))])
    upper = np.concatenate([np.full(n_modes, np.inf), np.full(n_modes, np.log(1e10))])
    res = least_squares(residuals_fn, x0, args=(omega, Gp, Gpp, n_modes),
                        bounds=(lower, upper), max_nfev=maxfev)
    # Estimate uncertainties from Jacobian
    J = res.jac
    dof = max(1, len(res.fun) - len(res.x))
    mse = np.sum(res.fun ** 2) / dof
    try:
        cov = np.linalg.inv(J.T @ J) * mse
        perr = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        perr = np.full_like(res.x, np.nan)

    Gs, taus = pack_params(res.x, n_modes)
    params = []
    for i in range(n_modes):
        err_G = perr[i]
        err_logtau = perr[n_modes + i]
        err_tau = np.exp(np.log(taus[i])) * err_logtau if not np.isnan(err_logtau) else np.nan
        params.append({
            "mode": i + 1,
            "G_k": Gs[i],
            "err_G_k": err_G,
            "tau_k": taus[i],
            "err_tau_k": err_tau
        })
    return res, pd.DataFrame(params)


# ------------------ Main program ------------------
if os.path.exists(filename):
    omega, Gp, Gpp = read_data(filename)
    print(f"Data source: {filename}")
else:
    # Generate synthetic demo data if no file found
    print("No data.txt found — generating synthetic example data...")
    omega = np.logspace(-1, 3, 40)
    true_Gs = np.array([1.2e3, 2.5e4])
    true_taus = np.array([0.02, 5.0])
    Gp_true, Gpp_true = generalized_maxwell_response(omega, true_Gs, true_taus)
    rng = np.random.default_rng(1)
    Gp = Gp_true * (1 + 0.05 * rng.standard_normal(len(omega)))
    Gpp = Gpp_true * (1 + 0.05 * rng.standard_normal(len(omega)))
    pd.DataFrame({"omega": omega, "Gp": Gp, "Gpp": Gpp}).to_csv("data_example.txt", index=False)
    print("Synthetic data saved as data_example.txt")

# --- Apply frequency range filter if requested ---
mask = np.ones_like(omega, dtype=bool)
if omega_min is not None:
    mask &= omega >= omega_min
if omega_max is not None:
    mask &= omega <= omega_max

omega_f, Gp_f, Gpp_f = omega[mask], Gp[mask], Gpp[mask]
if len(omega_f) < 3:
    raise ValueError("Not enough points in selected frequency range for fitting.")
if (omega_min is not None) or (omega_max is not None):
    print(f"Using {len(omega_f)} points in frequency range:", end=" ")
    print(f"{omega_min or min(omega):.3g} ≤ ω ≤ {omega_max or max(omega):.3g}")

# Perform fit
res, df_params = fit_generalized_maxwell(omega_f, Gp_f, Gpp_f, n_modes)

gof = compute_goodness_of_fit(omega_f, Gp_f, Gpp_f, res, n_modes)

print("\nGoodness of fit:")
for k, v in gof.items():
    print(f"{k}: {v:.5g}")

pd.DataFrame([gof]).to_csv(f"fit_{filename}_goodness.csv", index=False)

print("\nFitting summary:")
print("  Success:", res.success)
print("  Message:", res.message)
print("  Cost:", res.cost)
print("\nFitted parameters:")
print(df_params.round(5))

# Save results
df_params.to_csv(f"fit_{filename}_parameters.csv", index=False)
print(f"\nFitted parameters saved to fit_{filename}_parameters.csv")

# Plot
omega_smooth = np.logspace(np.log10(min(omega_f)), np.log10(max(omega_f)), 200)
Gp_fit, Gpp_fit = generalized_maxwell_response(
    omega_smooth,
    df_params["G_k"].values,
    df_params["tau_k"].values
)

plt.figure(figsize=(7, 5))
plt.loglog(omega, Gp, "o", color="gray", alpha=0.4, label="G' all data")
plt.loglog(omega, Gpp, "s", color="lightgray", alpha=0.4, label="G'' all data")
plt.loglog(omega_f, Gp_f, "o", label="G' fit range")
plt.loglog(omega_f, Gpp_f, "s", label="G'' fit range")
plt.loglog(omega_smooth, Gp_fit, "-", label="G' fit")
plt.loglog(omega_smooth, Gpp_fit, "--", label="G'' fit")
plt.xlabel("ω (rad/s)")
plt.ylabel("G', G'' (Pa)")
plt.title(f"Generalized Maxwell fit ({n_modes} modes)")
plt.legend()
plt.grid(True, which="both", ls=":", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{filename}_G'_G''_graph.png")
plt.show()
