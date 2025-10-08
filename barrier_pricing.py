import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from math import erf
import seaborn as sns
from numba import njit

sns.set_style('darkgrid')

#efficient version that doesnt store paths
def mc_barrier_price_delta(S0, K, B, r, sigma, T, N, M, reuse):
    dt = T / N
    sqrt_dt = np.sqrt(dt)
    logB = np.log(B)

    def _process_batch(Z_batch, U_batch):
        # previous and current log-price
        logS_prev = np.full(M, np.log(S0), dtype=np.float64)
        logS_curr = logS_prev.copy()
        alive = np.ones(M, dtype=bool)

        # time stepping
        for t in range(N):
            logS_curr = logS_prev + (r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * Z_batch[:, t]

            # deterministic exit
            ko_end = (logS_curr >= logB)
            alive &= ~ko_end

            # Exit probability method
            both_below = alive & (logS_prev < logB) & (logS_curr < logB)
            if np.any(both_below):
                p_hit = np.exp(
                    -2.0 * (logB - logS_prev[both_below]) * (logB - logS_curr[both_below]) / (sigma**2 * dt)
                )
                ko_bb = U_batch[both_below, t] < p_hit

                # flip those that randomly crossed
                idx = np.where(both_below)[0]
                alive[idx[ko_bb]] = False

            # advance
            logS_prev = logS_curr

        ST = np.exp(logS_curr)
        payoff = np.where(alive, np.maximum(ST - K, 0.0), 0.0)

        # Pathwise delta: 1_{alive} 1_{ST>K} * (ST/S0)
        indicator_ITM = (ST > K)
        delta_pw = alive & indicator_ITM

        # mean over paths
        batch_price = np.exp(-r * T) * payoff.mean()
        batch_delta = np.exp(-r * T) * np.mean(delta_pw * (ST / S0))

        return batch_price, batch_delta

    Z_all, U_all = reuse
    price, delta = _process_batch(Z_all, U_all)
    return price, delta

def _N(x):  # standard normal CDF
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

def bs_exact(S, K, B, r, sigma, T):

    if B <= S or T <= 0 or sigma <= 0:
        return 0.0

    volT = sigma * np.sqrt(T)
    mu = (r - 0.5 * sigma * sigma) / (sigma * sigma)
    a = (B / S) ** (2.0 * mu)
    b = (B / S) ** (2.0 * mu + 2.0)

    d1 = (np.log(S / K)   + (r + 0.5*sigma*sigma) * T) / volT
    d2 = (np.log(S / K)   + (r - 0.5*sigma*sigma) * T) / volT
    d3 = (np.log(S / B)   + (r + 0.5*sigma*sigma) * T) / volT
    d4 = (np.log(S / B)   + (r - 0.5*sigma*sigma) * T) / volT
    d5 = (np.log(S / B)   - (r - 0.5*sigma*sigma) * T) / volT
    d6 = (np.log(S / B)   - (r + 0.5*sigma*sigma) * T) / volT
    d7 = (np.log(S*K / (B*B)) - (r - 0.5*sigma*sigma) * T) / volT
    d8 = (np.log(S*K / (B*B)) - (r + 0.5*sigma*sigma) * T) / volT

    Se = S
    Ke = K * np.exp(-r * T)

    price = Se * (_N(d1) - _N(d3) - b * (_N(d6) - _N(d8))) \
          - Ke * (_N(d2) - _N(d4) - a * (_N(d5) - _N(d7)))
    return price

def mc_exotic_option(S0, K, B, r, sigma, T, N, M, cond_exit="True"):

    dt = T / N
    logB = np.log(B)

    # Simulate log-price paths
    logS = np.zeros((M, N + 1))
    logS[:, 0] = np.log(S0)

    Z = np.random.randn(M, N)
    U = np.random.rand(M, N)        
    alive = np.ones(M, dtype=bool)


    for t in range(N):
        logS[:, t+1] = logS[:, t] + (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t]

        # compute conditional exit probability only for alive paths
        for i in range(M):
            if not alive[i]:
                continue

            s0 = logS[i, t]
            s1 = logS[i, t+1]

            # if already crossed the barrier
            if s1 >= logB:
                alive[i] = False
            elif cond_exit:
                # both below barrier
                p_hit = np.exp(-2 * (logB - s0) * (logB - s1) / (sigma**2 * dt))
                if U[i, t] < p_hit:
                    alive[i] = False

    # convert back to normal prices
    S = np.exp(logS)
    ST = S[:, -1]

    # payoff
    payoff = np.where(alive, np.maximum(ST - K, 0.0), 0.0)
    price = np.exp(-r * T) * np.mean(payoff)

    # pathwise delta
    indicator_ITM = (ST > K).astype(float)
    delta_pathwise = np.exp(-r * T) * np.mean(alive * indicator_ITM * (ST / S0))

    return price, delta_pathwise, S

def paths_plotter(S, B, K, T):
    M, N_plus_1 = S.shape
    N = N_plus_1 - 1
    time_grid = np.linspace(0, T, N_plus_1)

    plt.figure(figsize=(8, 4))
    
    # Plot only a subset of paths to avoid clutter
    n_show = min(200, M)
    for i in range(n_show):
        if S[i, -1] < K:
            plt.plot(time_grid, S[i], lw=0.5, alpha=0.4, color='grey')
        elif S[i, -1] < B:
            plt.plot(time_grid, S[i], lw=0.6, alpha=0.4, color='blue')
        else:
            plt.plot(time_grid, S[i], lw=0.5, alpha=0.4, color='red')

    # Barrier line
    plt.axhline(y=B, color='r', linestyle='--', lw=1.2, label=f'Barrier = {B}')
    plt.axhline(y=K, color='b', linestyle='--', lw=1.2, label=f'Strike price = {K}')

    plt.title("Simulated Stock Price Paths (GBM)")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.grid(True, ls='--', alpha=0.6)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def vol_heatmap(K=100, B=150, r=0.05, T=1, N=250, M=10_000,
                S0_min=60, S0_max=120, S0_pts=50,
                sigma_min=0.2, sigma_max=1, sigma_pts=50, seed=42,
                savepath='delta_surface.png'):

    S0_vals    = np.linspace(S0_min, S0_max, S0_pts)
    sigma_vals = np.linspace(sigma_min, sigma_max, sigma_pts)

    # mesh for plotting
    s_mesh, sigma_mesh = np.meshgrid(S0_vals, sigma_vals, indexing='xy')

    # reuse random numbers
    rng = np.random.default_rng(seed)
    Z_reuse = rng.standard_normal((M, N))
    U_reuse = rng.random((M, N))

    price_grid = np.zeros_like(sigma_mesh, dtype=float)
    delta_grid = np.zeros_like(sigma_mesh, dtype=float)

    # fill surface
    for i, sig in enumerate(sigma_vals):
        print(f"Done with {i+1}/{len(sigma_vals)}")
        for j, s0 in enumerate(S0_vals):
            price, delta = mc_barrier_price_delta(s0, K, B, r, sig, T, N, M, reuse=(Z_reuse, U_reuse)
            )
            price_grid[i, j] = price
            delta_grid[i, j] = delta

    # 3D plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(s_mesh, sigma_mesh, delta_grid, cmap=cm.coolwarm, linewidth=0, antialiased=True)

    ax.set_title('Up-and-Out Call Delta Surface')
    ax.set_xlabel(r'Initial Stock Price $S_0$')
    ax.set_ylabel(r'Volatility $\sigma$')
    ax.set_zlabel('Option Price')

    fig.colorbar(surf, shrink=0.65, aspect=12, pad=0.08, label='Price')

    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.show()

    return s_mesh, sigma_mesh, price_grid, delta_grid



def bias_comparisson():
    S0 = 13
    K = 15
    B = 17
    r = 0.05
    sigma = 0.2
    T = 1.0
    M = 100_000
    N_vals = [5, 10, 20, 40, 80, 160, 320]

    bias_naive = []
    bias_cond_exit = []

    theo = bs_exact(S=S0, K=K, B=B, r=r, sigma=sigma, T=T)

    for N in N_vals:
        price_naive, delta, S = mc_exotic_option(S0, K, B, r, sigma, T, N, M, cond_exit=False)
        price_cond_exit, delta, S = mc_exotic_option(S0, K, B, r, sigma, T, N, M)

        bias_naive += [np.abs(price_naive-theo)]
        bias_cond_exit += [np.abs(price_cond_exit-theo)]

        print(f"Done with N={N}")

    a1, b1 = np.polyfit(np.log(N_vals), np.log(bias_naive), deg = 1)
    a2, b2 = np.polyfit(np.log(N_vals), np.log(bias_cond_exit), deg = 1)


    plt.figure()

    plt.plot(N_vals, bias_naive, label="Naive discretisation")
    plt.plot(N_vals, np.exp(a1*np.log(N_vals)+b1), label=f"y=ax+b: a={a1}, b={b1}")

    plt.plot(N_vals, bias_cond_exit, label="Conditional Exit")
    plt.plot(N_vals, np.exp(a2*np.log(N_vals)+b2), label=f"y=ax+b: a={a2}, b={b2}")

    plt.title("Bias comparison of naive discretisation and the conditional exit method")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()


def main():
    S0 = 20
    K = 20
    B = 25
    r = 0.05
    sigma = 1
    T = 1.0
    N = 250
    M = 100_000

    # # Calculate the price and delta
    # price, delta, S = mc_exotic_option(S0, K, B, r, sigma, T, N, M)
    # theo = bs_exact(S=S0, K=K, B=B, r=r, sigma=sigma, T=T)
    # print(f"Option price: {price:.4f}")
    # print(f"Theoretical price: {theo:.4f}")
    # print(f"Price error: {(price - theo):.4f}")
    # print(f"Delta: {delta:.4f}")

    # # Plot paths
    # paths_plotter(S=S, B=B, K=K, T=T)

    # Plot bias comparisson between cond. exit and naive discrete
    #bias_comparisson()

    vol_heatmap()



if __name__ == "__main__":
    main()