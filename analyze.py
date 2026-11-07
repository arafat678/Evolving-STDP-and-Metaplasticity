import json, glob, numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from snn_cl import *
from run_experiments import hypervolume, pareto_idx, REF

M = ["NSGA-II", "Random", "CMA-ES", "TPE-MO", "Hand-tuned"]
res = {m: [json.load(open(f)) for f in sorted(glob.glob(f"results/{m}_G1_run*.json"))] for m in M}
g2 = [json.load(open(f)) for f in sorted(glob.glob("results/NSGA-II_G2_run*.json"))]
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

def hv_of(pts): return hypervolume(np.array(pts)) if len(pts) else 0.0
def nd(pts):
    pts = np.array(pts); return pts[pareto_idx(pts)]

tab = {}
for m in M:
    hs = [hv_of(nd(r["search_front"])) for r in res[m]]
    hf = [hv_of(nd(r["fresh"])) for r in res[m]]
    hx = [hv_of(nd(r["shift"])) for r in res[m]]
    tab[m] = dict(search=hs, fresh=hf, shift=hx)
if g2:
    tab["NSGA-II (G2)"] = dict(search=[hv_of(nd(r["search_front"])) for r in g2], fresh=[hv_of(nd(r["fresh"])) for r in g2], shift=[hv_of(nd(r["shift"])) for r in g2])

P("Hypervolume (ref = chance 0.5,0.5), mean ± sd over independent optimizer runs")
P(f"{'method':14s} n  HV_search(noisy)   HV_fresh(unseen seeds/orders)   HV_shifted-domain   transfer gap(fresh-shift)")
for m, d in tab.items():
    f = lambda x: f"{np.mean(x):.4f}±{np.std(x, ddof=1) if len(x)>1 else 0:.4f}"
    gap = np.mean(d['fresh']) - np.mean(d['shift'])
    P(f"{m:14s} {len(d['fresh'])}  {f(d['search'])}    {f(d['fresh'])}          {f(d['shift'])}      {gap:.4f}")

def cohen(a, b):
    sp = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2); return (np.mean(a) - np.mean(b)) / sp if sp > 0 else float('nan')
def boot(a, b, n=5000):
    rng = np.random.default_rng(0); d = [np.mean(rng.choice(a, len(a))) - np.mean(rng.choice(b, len(b))) for _ in range(n)]
    return np.percentile(d, [2.5, 97.5])
P("\nNSGA-II vs baselines on held-out fresh HV (one-sided Mann-Whitney U, H1: NSGA-II > baseline)")
for m in ["Random", "CMA-ES", "TPE-MO", "Hand-tuned"]:
    a, b = tab["NSGA-II"]["fresh"], tab[m]["fresh"]
    u = stats.mannwhitneyu(a, b, alternative="greater")
    lo, hi = boot(a, b)
    P(f"  vs {m:10s} dHV={np.mean(a)-np.mean(b):+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  p={u.pvalue:.4f}  d={cohen(a,b):+.2f}")

# ---- CL metrics of best-balanced held-out solution (max of sum) per method
P("\nContinual-learning metrics of the 'balanced' solution (max f_pl+f_st on held-out seeds), mean over runs")
for m in M:
    accs = []
    for r in res[m]:
        fr = np.array(r["fresh"]); i = fr.sum(1).argmax(); accs.append(fr[i])
    accs = np.array(accs).mean(0); P(f"  {m:11s} plasticity={accs[0]:.3f} stability={accs[1]:.3f}")

# ---- design rules (C3)
arch = np.array([a[0] for r in res["NSGA-II"] for a in r["archive"]])
fs = np.array([[a[1], a[2]] for r in res["NSGA-II"] for a in r["archive"]])
from sklearn.ensemble import RandomForestRegressor
score = fs.sum(1)
rf = RandomForestRegressor(300, random_state=0).fit(arch, score)
imp = rf.feature_importances_
P("\nGene importance (random-forest on all evaluated NSGA-II genomes; fANOVA proxy)")
for (n, *_), v in sorted(zip(GENES, imp), key=lambda t: -t[1]): P(f"  {n:8s} {v:.3f}")
fronts = [np.array(z) for r in res["NSGA-II"] for z in r["z"]]
fronts = np.array(fronts)
P(f"\nFraction of final NSGA-II Pareto solutions with metaplasticity on (m>0): {np.mean(fronts[:,4] >= -1.5):.2f}  (n={len(fronts)})")
P(f"Fraction in random-search pool with m>0 (prior): {np.mean((np.array([a[0] for r in res['Random'] for a in r['archive']])[:,4])>=-1.5):.2f}")
P("Median evolved Pareto genome (log10 units for log genes):")
for (n, *_), v in zip(GENES, np.median(fronts, 0)): P(f"  {n:8s} {v:+.2f}")
c = np.corrcoef(fronts[:, 1], fronts[:, 4])[0, 1]; P(f"corr(log rho, log meta) on evolved fronts = {c:+.2f}")

# ---- ablation on evolved best genome, fresh seeds
best = res["NSGA-II"]
cands = [(np.array(z), np.array(f)) for r in best for z, f in zip(r["z"], r["fresh"])]
zb = max(cands, key=lambda t: t[1].sum())[0]
def ev(z, gate="G1", n=10):
    a = [run_sequence(z, random_order(np.random.default_rng(7000 + s)), 7000 + s, gate) for s in range(n)]
    o = np.array([objectives(x) for x in a]); c_ = np.array([cl_metrics(x) for x in a])
    return o.mean(0), o.std(0), c_.mean(0)
P("\nAblation of best evolved genome (10 fresh seeds/orders):  plasticity, stability | ACC, BWT, Forgetting")
abl = {}
z0 = zb.copy(); abl["full (G1)"] = ev(zb)
z1 = zb.copy(); z1[4] = -2.0; abl["no metaplasticity"] = ev(z1)
z2 = zb.copy(); z2[8] = -6.0; abl["no homeostasis (gamma~0)"] = ev(z2)
z3 = zb.copy(); z3[4] = -2.0; z3[8] = -6.0; abl["STDP only"] = ev(z3)
abl["full, gate G2 (same genome)"] = ev(zb, "G2")
abl["hand-tuned STDP"] = ev(HAND)
for k, (m_, s_, c_) in abl.items():
    P(f"  {k:28s} pl={m_[0]:.3f}±{s_[0]:.3f} st={m_[1]:.3f}±{s_[1]:.3f} | ACC={c_[0]:.3f} BWT={c_[1]:+.3f} F={c_[2]:.3f}")
a_, ex = run_sequence(zb, random_order(np.random.default_rng(7000)), 7000, return_extra=True)
P(f"saturated-synapse fraction (W>0.9 w_max) of best genome: {ex['sat']:.3f}")
P("best genome:", {k: round(v, 4) for k, v in decode(zb).items()})
json.dump(dict(zb=zb.tolist()), open("results/best_genome.json", "w"))

# ---- plots
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
col = dict(zip(M, ["C3", "C7", "C0", "C2", "k"]))
for m in M:
    for j, r in enumerate(res[m]):
        P_ = nd(r["fresh"]); P_ = P_[np.argsort(P_[:, 0])]
        ax[0].plot(P_[:, 0], P_[:, 1], "-o", c=col[m], alpha=.35, ms=3, label=m if j == 0 else None)
ax[0].set_xlabel("plasticity f_pl"); ax[0].set_ylabel("stability f_st"); ax[0].set_title("Held-out Pareto fronts (all runs)"); ax[0].legend(fontsize=8)
pos = np.arange(len(tab)); names = list(tab)
ax[1].bar(pos - .2, [np.mean(tab[m]["fresh"]) for m in names], .4, yerr=[np.std(tab[m]["fresh"]) for m in names], label="unseen orders")
ax[1].bar(pos + .2, [np.mean(tab[m]["shift"]) for m in names], .4, yerr=[np.std(tab[m]["shift"]) for m in names], label="shifted domain")
ax[1].set_xticks(pos); ax[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8); ax[1].set_ylabel("hypervolume"); ax[1].legend(); ax[1].set_title("Hypervolume (C1) and transfer (C2)")
o = np.argsort(-imp); ax[2].barh([GENES[i][0] for i in o][::-1], imp[o][::-1]); ax[2].set_title("Gene importance (C3)")
plt.tight_layout(); plt.savefig("results_fig1.png", dpi=140)

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for k, (nm, z, g) in enumerate([("Evolved (G1)", zb, "G1"), ("STDP only", z3, "G1"), ("Hand-tuned", HAND, "G1")]):
    A = np.mean([run_sequence(z, random_order(np.random.default_rng(7000 + s)), 7000 + s, g) for s in range(5)], 0)
    im = ax[k].imshow(A, vmin=.4, vmax=1, cmap="viridis"); ax[k].set_title(f"a(t,i): {nm}"); ax[k].set_xlabel("task i"); ax[k].set_ylabel("after training task t")
    for t in range(5):
        for i in range(t + 1): ax[k].text(i, t, f"{A[t,i]:.2f}", ha="center", va="center", color="w", fontsize=8)
plt.tight_layout(); plt.savefig("results_fig2.png", dpi=140)
sc = plt.figure(figsize=(5.5, 4.5)); 
plt.scatter(arch[:, 1], arch[:, 4], c=score, cmap="viridis", s=14); plt.colorbar(label="f_pl+f_st")
plt.xlabel("log10 rho (LTD/LTP)"); plt.ylabel("log10 metaplasticity m"); plt.title("Landscape, all NSGA-II genomes"); plt.tight_layout(); plt.savefig("results_fig3.png", dpi=140)
open("results_summary.txt", "w").write("\n".join(out))
