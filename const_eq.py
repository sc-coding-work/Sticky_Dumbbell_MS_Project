import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Define parameters (set your own values)
k_plus = 1.0
k_minus = 0.5
tau_u = 1.0
tau_b = 10.0
nkT=16877.04 # pascal    
#α       τb     τu     k−     k+
# 1.1  2.4487 0.0378 1.3130 0.62226
# 2.0  5.5923 0.0379 1.5544 0.7183
#10.0 57.3744 0.0381 1.7256 0.7830


# Peterlin function
def peterlin(sigma, L2=50.0):
    tr_sigma = np.trace(sigma)
    return (L2 - 3.0) / (L2 - tr_sigma)
    
# Velocity gradient tensor (example: simple shear)
def grad_v(t):
    return np.array([[0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0]])


# Upper-convected derivative
# ∇σ = dσ/dt - (σ·∇v + (∇v)^T·σ)
def upper_convected(t, sigma, grad):
    return - (sigma @ grad + grad.T @ sigma)


# System of ODEs for sigma_u and sigma_b (both 3x3 -> flattened 18 variables)
def rhs(t, y):
    sigma_u = y[:9].reshape((3, 3))
    sigma_b = y[9:18].reshape((3, 3))
    p = y[18]
    grad = grad_v(t)
    
    # Compute Peterlin factors
    fP_u = peterlin(sigma_u)
    fP_b = peterlin(sigma_b)

    # Eqn for sigma_u
    dsigma_u_dt = (1-p)* nkT * (grad + grad.T) \
                   - ((1 + k_plus * tau_u) * fP_u * sigma_u \
                   + k_minus * tau_u * fP_b * sigma_b \
                   - tau_u * upper_convected(t, sigma_u, grad))/tau_u

    # Eqn for sigma_b
    dsigma_b_dt = p* nkT * (grad + grad.T) \
                   - ((1 + k_minus * tau_b) * fP_b * sigma_b \
                   + k_plus * tau_b * fP_u * sigma_u \
                   - tau_b * upper_convected(t, sigma_b, grad))/tau_b
                   
    dp_dt = k_plus * (1-p ) - k_minus * p

    return np.concatenate([dsigma_u_dt.flatten(), dsigma_b_dt.flatten(), [dp_dt]])


# Initial conditions (identity tensors)
sigma_u0 = np.eye(3)
sigma_b0 = np.eye(3)
p0 = 1.0/( 1.0 + k_plus/k_minus ) # fraction of closed stickers
y0 = np.concatenate([sigma_u0.flatten(), sigma_b0.flatten(), [p0]])


# Time span
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 200)


# Solve
sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method='RK45')


# Extract solutions
sigma_u_sol = sol.y[:9, :].T.reshape(-1, 3, 3)
sigma_b_sol = sol.y[9:18, :].T.reshape(-1, 3, 3)
p_sol = sol.y[18:, :] # extract p solution


# Example: print sigma_u at final time
print("Sigma_u at t=10:")
print(sigma_u_sol[-1])

# ---- Plotting ----
t = sol.t

fig, axs = plt.subplots(2, 1, figsize=(8, 10))

# Normal stresses (xx, yy, zz from both sigma_u and sigma_b)
axs[0].plot(t, sigma_u_sol[:, 0, 0], label=r"$\sigma_{u,xx}$")
axs[0].plot(t, sigma_u_sol[:, 1, 1], label=r"$\sigma_{u,yy}$")
axs[0].plot(t, sigma_u_sol[:, 2, 2], label=r"$\sigma_{u,zz}$")
axs[0].plot(t, sigma_b_sol[:, 0, 0], '--', label=r"$\sigma_{b,xx}$")
axs[0].plot(t, sigma_b_sol[:, 1, 1], '--', label=r"$\sigma_{b,yy}$")
axs[0].plot(t, sigma_b_sol[:, 2, 2], '--', label=r"$\sigma_{b,zz}$")
axs[0].set_title("Normal stresses")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Stress")
axs[0].legend()
axs[0].grid(True)

# Shear stresses (xy from both sigma_u and sigma_b, you can add yz, xz too)
axs[1].plot(t, sigma_u_sol[:, 0, 1], label=r"$\sigma_{u,xy}$")
axs[1].plot(t, sigma_u_sol[:, 1, 0], label=r"$\sigma_{u,yx}$")
axs[1].plot(t, sigma_b_sol[:, 0, 1], '--', label=r"$\sigma_{b,xy}$")
axs[1].plot(t, sigma_b_sol[:, 1, 0], '--', label=r"$\sigma_{b,yx}$")
axs[1].set_title("Shear stresses")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Stress")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

