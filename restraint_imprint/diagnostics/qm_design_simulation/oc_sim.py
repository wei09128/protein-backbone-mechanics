"""Operating characteristics of the confirmatory-QM decision rule (single angle).
q_ij (centred QM) = s_i * v_ij + e_ij, centred within residue; s_i = s + eta_i (residue heterogeneity);
e_ij noise whose size is set by the target correlation r(QM, v3). Estimator: through-origin slope pooled
over residues; residue-clustered percentile bootstrap. Decision: v3 supported if CI includes 1 and excludes rho;
v2 supported if CI includes rho and excludes 1; gate r_hat < 0.3 -> 'not reproduced'."""
import numpy as np, itertools, sys
rng = np.random.default_rng(20261117)
def one(G, k, s, r, tau, rho=0.6, sdL=0.5, B=2000, alpha=0.05):
    v = rng.normal(0, sdL, (G, k)); v -= v.mean(1, keepdims=True)
    si = s + rng.normal(0, tau, G)
    se = s * sdL * np.sqrt(max(1 / r**2 - 1, 0))
    q = si[:, None] * v + rng.normal(0, se, (G, k)); q -= q.mean(1, keepdims=True)
    sxy, sxx = (q * v).sum(1), (v * v).sum(1)
    shat = sxy.sum() / sxx.sum()
    rhat = np.corrcoef(q.ravel(), v.ravel())[0, 1]
    W = rng.multinomial(G, np.ones(G) / G, size=B)
    bs = (W @ sxy) / (W @ sxx)
    lo, hi = np.quantile(bs, [alpha / 2, 1 - alpha / 2])
    cover = lo <= s <= hi
    if rhat < 0.3: return 'gate', cover, hi - lo
    inc1, incr = lo <= 1 <= hi, lo <= rho <= hi
    if inc1 and not incr: return 'v3', cover, hi - lo
    if incr and not inc1: return 'v2', cover, hi - lo
    return 'inc', cover, hi - lo
R = 400
print('G  k  s_true  r   tau  alpha | P(v3)  P(v2)  P(inconcl) P(gate) | coverage  mean_CI_width')
for G, k, s, r, tau, a in itertools.product([20, 12], [4, 8], [0.6, 1.0], [0.5, 0.75], [0.0, 0.2], [0.05, 0.05 / 3]):
    out = [one(G, k, s, r, tau, alpha=a) for _ in range(R)]
    d = [o[0] for o in out]
    print(f"{G:2d} {k}  {s:.1f}   {r:.2f} {tau:.1f} {a:.3f} | {d.count('v3')/R:5.2f}  {d.count('v2')/R:5.2f}  {d.count('inc')/R:6.2f}   {d.count('gate')/R:5.2f}  |"
          f"  {np.mean([o[1] for o in out]):.3f}    {np.mean([o[2] for o in out]):.2f}", flush=True)
