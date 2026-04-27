import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.stats import rice, chi2, ncx2, norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'lines.linewidth': 1.0,
})

OUT = Path('/mnt/data/power_probability_python_outputs')
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260414
FS = 10_000
F0 = 50
T_TOTAL = 1.0
N = int(FS * T_TOTAL)
TS = 1 / FS
T = np.arange(N) / FS
CYCLE = FS // F0
SNR_LIST = np.array([30, 20, 15, 10, 5], dtype=float)
MC_DESC = 250
MC_MSE = 300
MC_PROB = 5000
rng = np.random.default_rng(SEED)

A0 = 1.0
HARM_AMP = {5: 0.05, 7: 0.03}
HARM_PHI = {5: 0.4, 7: -0.8}
TRANSIENT_AMP = 0.20
TRANSIENT_FREQ = 600
TRANSIENT_DECAY = 120
TRANSIENT_T0 = 0.62
RHO_L = 0.9995
STD_L = 0.006
RHO_R = 0.9992
STD_R = 0.008
GAMMA_R = 0.8
ALPHA_FA = 0.05
EPS_AMP = 0.01


def ar1_process(n: int, rho: float, std_target: float, rng_local: np.random.Generator) -> np.ndarray:
    sigma_w = std_target * math.sqrt(1 - rho ** 2)
    w = rng_local.normal(scale=sigma_w, size=n)
    x = np.zeros(n)
    for k in range(1, n):
        x[k] = rho * x[k - 1] + w[k]
    x -= x.mean()
    return x


def noise_sigma_for_snr(snr_db: float) -> float:
    return math.sqrt(0.5 / (10 ** (snr_db / 10)))


def build_deterministic_components():
    s0 = A0 * np.sin(2 * np.pi * F0 * T)
    harm = np.zeros_like(s0)
    for h, amp in HARM_AMP.items():
        harm += amp * np.sin(2 * np.pi * h * F0 * T + HARM_PHI[h])
    u = (T >= TRANSIENT_T0).astype(float)
    tau = TRANSIENT_AMP * np.exp(-TRANSIENT_DECAY * (T - TRANSIENT_T0)) * np.sin(2 * np.pi * TRANSIENT_FREQ * (T - TRANSIENT_T0)) * u
    return s0, harm, tau


def slow_random_component(rng_local: np.random.Generator):
    L = ar1_process(N, RHO_L, STD_L, rng_local)
    R = ar1_process(N, RHO_R, STD_R, rng_local)
    delta = (L - GAMMA_R * R) * S0
    return L, R, delta


def one_cycle_corr(x: np.ndarray, cycle_samples: int = CYCLE) -> float:
    x0 = x - x.mean()
    return float(np.dot(x0[:-cycle_samples], x0[cycle_samples:]) / ((len(x0) - cycle_samples) * np.var(x0, ddof=0)))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


S0, HARM, TAU = build_deterministic_components()
P_SIGNAL = float(np.mean(S0**2))
P_HARM = float(np.mean(HARM**2))
P_TAU = float(np.mean(TAU**2))
P_DELTA_THEORY = 0.5 * (STD_L**2 + (GAMMA_R**2) * STD_R**2)
MSE_FLOOR_THEORY = P_HARM + P_TAU + P_DELTA_THEORY

# Representative realization
rep_rng = np.random.default_rng(101)
_, _, DELTA_REP = slow_random_component(rep_rng)
N20 = rep_rng.normal(scale=noise_sigma_for_snr(20), size=N)
X20_FULL = S0 + HARM + TAU + DELTA_REP + N20

# Fig 1 waveform
mask = (T >= 0.585) & (T <= 0.675)
fig = plt.figure(figsize=(3.25, 2.05))
plt.plot(T[mask], S0[mask], '--', linewidth=1.0)
plt.plot(T[mask], X20_FULL[mask], linewidth=1.2)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (pu)')
plt.tight_layout(pad=0.4)
fig.savefig(OUT / 'fig1_waveform.png', dpi=300)
plt.close(fig)

# Fig 2 histogram
fig = plt.figure(figsize=(3.25, 2.10))
mu_n = float(np.mean(N20))
std_n = float(np.std(N20, ddof=1))
skew_n = float(stats.skew(N20, bias=False))
kurt_n = float(stats.kurtosis(N20, fisher=True, bias=False))
_, ks_p = stats.kstest((N20 - mu_n) / std_n, 'norm')
counts, bins, _ = plt.hist(N20, bins=48, density=True, alpha=0.65, label='Empirical noise')
xpdf = np.linspace(bins.min(), bins.max(), 600)
pdf = (1.0 / (std_n * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xpdf - mu_n) / std_n) ** 2)
plt.plot(xpdf, pdf, linewidth=1.6, label='Gaussian fit')
plt.xlabel('Noise sample value')
plt.ylabel('Density')
plt.text(0.98, 0.95, f'mean={mu_n:.4f}\nstd={std_n:.4f}\nskew={skew_n:.3f}\nex.kurt={kurt_n:.3f}\nKS p={ks_p:.3f}',
         transform=plt.gca().transAxes, ha='right', va='top', fontsize=7,
         bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85, edgecolor='0.7'))
plt.tight_layout(pad=0.4)
fig.savefig(OUT / 'fig2_noise_hist.png', dpi=300)
plt.close(fig)

# Fig 3 PSD
f_psd, S_psd = signal.welch(X20_FULL, fs=FS, nperseg=2048)
fig = plt.figure(figsize=(3.25, 2.10))
plt.semilogy(f_psd, S_psd, linewidth=1.1)
for fx, txt in [(50, '50 Hz'), (250, '5th'), (350, '7th'), (600, 'Transient')]:
    plt.axvline(fx, linestyle='--', linewidth=0.8)
    y = np.interp(fx, f_psd, S_psd)
    plt.text(fx + 8, y * 1.15, txt, fontsize=7)
plt.xlim(0, 800)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (pu^2/Hz)')
plt.tight_layout(pad=0.4)
fig.savefig(OUT / 'fig3_psd.png', dpi=300)
plt.close(fig)

# Table I
PARAMS = pd.DataFrame([
    ('Nominal frequency', '50 Hz'),
    ('Sampling frequency', '10 kHz'),
    ('Record length', '1 s (10,000 samples)'),
    ('Fundamental amplitude', '1.0 pu'),
    ('Harmonics', '5th = 5%, 7th = 3%'),
    ('Transient', '0.2 pu decaying 600 Hz oscillation from 0.62 s'),
    ('Load process', 'AR(1), rho = 0.9995, std = 0.006 pu'),
    ('Renewable process', 'AR(1), rho = 0.9992, std = 0.008 pu'),
    ('Monte Carlo trials', '250 (desc.), 300 (MSE), 5000 (prob.)'),
    ('False alarm target', 'alpha = 5%'),
], columns=['Parameter', 'Value'])
PARAMS.to_csv(OUT / 'table1_parameters.csv', index=False)

# Table II descriptive stats at 15 dB
rows = []
rng_desc = np.random.default_rng(222)
sigma15 = noise_sigma_for_snr(15)
for _ in range(MC_DESC):
    n = rng_desc.normal(scale=sigma15, size=N)
    _, _, delta = slow_random_component(rng_desc)
    cases = {
        'Clean fundamental': S0,
        'AWGN only': S0 + n,
        'AWGN + harmonics': S0 + HARM + n,
        'Full stochastic case': S0 + HARM + TAU + delta + n,
    }
    for name, x in cases.items():
        rows.append({
            'Case': name,
            'Variance': float(np.var(x, ddof=1)),
            'RMS': float(np.sqrt(np.mean(x**2))),
            'MSE': float(np.mean((x - S0) ** 2)),
            'One-cycle corr.': one_cycle_corr(x),
        })
DESC = pd.DataFrame(rows).groupby('Case', as_index=False).mean()
order = ['Clean fundamental', 'AWGN only', 'AWGN + harmonics', 'Full stochastic case']
DESC['Case'] = pd.Categorical(DESC['Case'], categories=order, ordered=True)
DESC = DESC.sort_values('Case').reset_index(drop=True)
DESC.to_csv(OUT / 'table2_descriptive_stats_15db.csv', index=False)

# MSE summary
rng_mse = np.random.default_rng(333)
mse_records = []
for snr_db in SNR_LIST:
    sigma = noise_sigma_for_snr(float(snr_db))
    vals = []
    for _ in range(MC_MSE):
        n = rng_mse.normal(scale=sigma, size=N)
        _, _, delta = slow_random_component(rng_mse)
        x = S0 + HARM + TAU + delta + n
        vals.append(np.mean((x - S0) ** 2))
    vals = np.asarray(vals)
    mean_emp = float(vals.mean())
    sd = float(vals.std(ddof=1))
    ci = 1.96 * sd / math.sqrt(MC_MSE)
    mse_records.append({
        'SNR_dB': float(snr_db),
        'Mean_MSE_emp': mean_emp,
        'CI95_low': mean_emp - ci,
        'CI95_high': mean_emp + ci,
        'Mean_MSE_theory': MSE_FLOOR_THEORY + sigma**2,
    })
MSE = pd.DataFrame(mse_records)
MSE.to_csv(OUT / 'table_mse_summary.csv', index=False)

fig = plt.figure(figsize=(3.25, 2.10))
plt.errorbar(MSE['SNR_dB'], MSE['Mean_MSE_emp'],
             yerr=[MSE['Mean_MSE_emp'] - MSE['CI95_low'], MSE['CI95_high'] - MSE['Mean_MSE_emp']],
             fmt='o-', linewidth=1.2, markersize=4, label='Empirical mean MSE')
plt.plot(MSE['SNR_dB'], MSE['Mean_MSE_theory'], '--', linewidth=1.3, label='Expectation model')
plt.xlabel('SNR (dB)')
plt.ylabel('Mean MSE (pu^2)')
plt.tight_layout(pad=0.4)
fig.savefig(OUT / 'fig4_mse_vs_snr.png', dpi=300)
plt.close(fig)

# Amplitude probability - vectorized
n_idx = np.arange(CYCLE) / FS
H_amp = np.column_stack([np.sin(2 * np.pi * F0 * n_idx), np.cos(2 * np.pi * F0 * n_idx)])
G_amp = np.linalg.inv(H_amp.T @ H_amp) @ H_amp.T
s_cycle = np.sin(2 * np.pi * F0 * n_idx)

amp_records = []
rng_amp = np.random.default_rng(444)
for snr_db in SNR_LIST:
    sigma = noise_sigma_for_snr(float(snr_db))
    noise = rng_amp.normal(scale=sigma, size=(CYCLE, MC_PROB))
    Y = s_cycle[:, None] + noise
    coeff = G_amp @ Y
    amp_hat = np.sqrt(coeff[0, :]**2 + coeff[1, :]**2)
    flag = np.abs(amp_hat - 1.0) > EPS_AMP
    emp = float(flag.mean())
    lo, hi = wilson_interval(int(flag.sum()), MC_PROB)
    sigma_theta = math.sqrt(2 * sigma**2 / CYCLE)
    b = 1.0 / sigma_theta
    theory = float(rice.cdf(1 - EPS_AMP, b=b, scale=sigma_theta) + 1 - rice.cdf(1 + EPS_AMP, b=b, scale=sigma_theta))
    amp_records.append({
        'SNR_dB': float(snr_db),
        'P_emp': emp,
        'P_theory': theory,
        'CI95_low': lo,
        'CI95_high': hi,
        'sigma_theta': sigma_theta,
    })
AMP = pd.DataFrame(amp_records)
AMP.to_csv(OUT / 'table_amp_probability.csv', index=False)

fig = plt.figure(figsize=(3.25, 2.10))
plt.errorbar(AMP['SNR_dB'], AMP['P_emp'],
             yerr=[AMP['P_emp'] - AMP['CI95_low'], AMP['CI95_high'] - AMP['P_emp']],
             fmt='o', markersize=4, linewidth=1.1, label='Empirical')
plt.plot(AMP['SNR_dB'], AMP['P_theory'], '--', linewidth=1.3, label='Rice-theory')
plt.xlabel('SNR (dB)')
plt.ylabel('P(|A_hat - A| > 1%)')
plt.ylim(-0.02, 1.02)
plt.tight_layout(pad=0.4)
fig.savefig(OUT / 'fig5_amp_probability.png', dpi=300)
plt.close(fig)

# Detector probability - vectorized
cols = []
for h in [1, 5, 7]:
    cols.append(np.sin(2 * np.pi * h * F0 * n_idx))
    cols.append(np.cos(2 * np.pi * h * F0 * n_idx))
H_det = np.column_stack(cols)
P_det = H_det @ np.linalg.inv(H_det.T @ H_det) @ H_det.T
M_det = np.eye(CYCLE) - P_det
nu = int(round(np.trace(M_det)))
start_idx = 80
u_step = (np.arange(CYCLE) >= start_idx).astype(float)
TAU_CYCLE = TRANSIENT_AMP * np.exp(-TRANSIENT_DECAY * (n_idx - n_idx[start_idx])) * np.sin(2 * np.pi * TRANSIENT_FREQ * (n_idx - n_idx[start_idx])) * u_step
r_tau = M_det @ TAU_CYCLE
h_cycle = 0.05 * np.sin(2 * np.pi * 5 * F0 * n_idx + 0.4) + 0.03 * np.sin(2 * np.pi * 7 * F0 * n_idx - 0.8)
base_cycle = s_cycle + h_cycle

det_records = []
rng_det = np.random.default_rng(555)
for snr_db in SNR_LIST:
    sigma = noise_sigma_for_snr(float(snr_db))
    eta = (sigma**2 / CYCLE) * chi2.ppf(1 - ALPHA_FA, df=nu)
    threshold_scaled = CYCLE * eta / (sigma**2)
    lambda_nc = float(np.dot(r_tau, r_tau) / (sigma**2))
    theory_pd = float(1 - ncx2.cdf(threshold_scaled, df=nu, nc=lambda_nc))
    noise = rng_det.normal(scale=sigma, size=(CYCLE, MC_PROB))
    X0 = base_cycle[:, None] + noise
    X1 = X0 + TAU_CYCLE[:, None]
    R0 = M_det @ X0
    R1 = M_det @ X1
    J0 = np.mean(R0**2, axis=0)
    J1 = np.mean(R1**2, axis=0)
    count_fa = int(np.sum(J0 > eta))
    count_d = int(np.sum(J1 > eta))
    p_fa = count_fa / MC_PROB
    p_d = count_d / MC_PROB
    lo, hi = wilson_interval(count_d, MC_PROB)
    det_records.append({
        'SNR_dB': float(snr_db),
        'eta': eta,
        'P_FA_emp': p_fa,
        'P_D_emp': p_d,
        'P_D_theory': theory_pd,
        'CI95_low': lo,
        'CI95_high': hi,
        'lambda': lambda_nc,
    })
DET = pd.DataFrame(det_records)
DET.to_csv(OUT / 'table_detector_probability.csv', index=False)

fig = plt.figure(figsize=(3.25, 2.10))
plt.errorbar(DET['SNR_dB'], DET['P_D_emp'],
             yerr=[DET['P_D_emp'] - DET['CI95_low'], DET['CI95_high'] - DET['P_D_emp']],
             fmt='s', markersize=4, linewidth=1.1, label='Empirical P_D')
plt.plot(DET['SNR_dB'], DET['P_D_theory'], '--', linewidth=1.3, label='Noncentral chi-square theory')
plt.xlabel('SNR (dB)')
plt.ylabel('Detection probability P_D')
plt.ylim(-0.02, 1.02)
plt.tight_layout(pad=0.4)
fig.savefig(OUT / 'fig6_detector_probability.png', dpi=300)
plt.close(fig)

SUMMARY = pd.DataFrame({
    'SNR_dB': SNR_LIST,
    'MSE_emp': MSE['Mean_MSE_emp'],
    'MSE_theory': MSE['Mean_MSE_theory'],
    'P_amp_emp': AMP['P_emp'],
    'P_amp_theory': AMP['P_theory'],
    'P_D_emp': DET['P_D_emp'],
    'P_D_theory': DET['P_D_theory'],
    'P_FA_emp': DET['P_FA_emp'],
})
SUMMARY.to_csv(OUT / 'table3_probability_summary.csv', index=False)

beta = 0.10
q = norm.ppf(1 - beta / 2)
snr_min_approx = 10 * math.log10((q ** 2) / (CYCLE * EPS_AMP ** 2))

results = {
    'noise_stats': {
        'mean': mu_n,
        'std': std_n,
        'skew': skew_n,
        'excess_kurtosis': kurt_n,
        'ks_p_value': float(ks_p),
    },
    'mse_floor_theory': MSE_FLOOR_THEORY,
    'mse_summary': MSE.to_dict(orient='records'),
    'amp_summary': AMP.to_dict(orient='records'),
    'det_summary': DET.to_dict(orient='records'),
    'design_rule_snr_min_for_beta_10pct': snr_min_approx,
    'nu_detector': nu,
    'power_terms': {
        'signal': P_SIGNAL,
        'harmonic': P_HARM,
        'transient': P_TAU,
        'delta_theory': P_DELTA_THEORY,
    },
}
with open(OUT / 'results_summary.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
with open(OUT / 'results_summary.txt', 'w', encoding='utf-8') as f:
    f.write('MSE summary\n')
    f.write(MSE.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    f.write('\n\nAmplitude probability summary\n')
    f.write(AMP.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    f.write('\n\nDetector summary\n')
    f.write(DET.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    f.write(f'\n\nApproximate minimum SNR for 10% exceedance risk = {snr_min_approx:.3f} dB\n')

print('[OK] built assets in', OUT)
print(MSE.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
print(AMP.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
print(DET.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
