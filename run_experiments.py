import sys, json, os, time
import numpy as np
from snn_cl import *

N, G, R = 12, 10, 3          # population, generations, seeds per genome  -> budget B = N*G = 120 evaluations
B = N * G
RUNS = int(os.environ.get("RUNS", 6))
OUT = "results"; os.makedirs(OUT, exist_ok=True)
REF = np.array([0.5, 0.5])   # chance level for both objectives

# ---------------- generic helpers ----------------
class Evaluator:
    def __init__(self, run, gate="G1"):
        self.run, self.gate, self.n = run, gate, 0
        self.archive = []   # (z, f_pl, f_st)
    def __call__(self, z):
        seeds = [self.run * 10000 + 3 * self.n + r for r in range(R)]
        f = fitness(z, seeds, self.gate); self.n += 1
        self.archive.append((list(map(float, z)), float(f[0]), float(f[1])))
        return f

def dominates(a, b): return np.all(a >= b) and np.any(a > b)

def nd_sort(F):
    n = len(F); S = [[] for _ in range(n)]; cnt = np.zeros(n, int); fronts = [[]]
    for p in range(n):
        for q in range(n):
            if dominates(F[p], F[q]): S[p].append(q)
            elif dominates(F[q], F[p]): cnt[p] += 1
        if cnt[p] == 0: fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                cnt[q] -= 1
                if cnt[q] == 0: nxt.append(q)
        i += 1; fronts.append(nxt)
    return fronts[:-1]

def crowding(F, idx):
    d = {i: 0.0 for i in idx}
    for m in range(F.shape[1]):
        s = sorted(idx, key=lambda i: F[i, m])
        d[s[0]] = d[s[-1]] = np.inf
        rng_ = F[s[-1], m] - F[s[0], m] + 1e-12
        for a, b, c in zip(s[:-2], s[1:-1], s[2:]): d[b] += (F[c, m] - F[a, m]) / rng_
    return d

def hypervolume(P, ref=REF):
    P = np.array([p for p in P if np.all(p > ref)])
    if len(P) == 0: return 0.0
    P = P[np.argsort(-P[:, 0])]; hv = 0.0; ymax = ref[1]
    for x, y in P:
        if y > ymax: hv += (x - ref[0]) * (y - ymax); ymax = y
    return hv

def pareto_idx(F):
    return nd_sort(np.array(F))[0]

# ---------------- optimizers ----------------
def nsga2(ev, rng):
    pop = rng.uniform(LO, HI, (N, D)); F = np.array([ev(z) for z in pop])
    for g in range(G - 1):
        fronts = nd_sort(F); rank = np.zeros(len(pop), int); cd = np.zeros(len(pop))
        for r, fr in enumerate(fronts):
            c = crowding(F, fr)
            for i in fr: rank[i] = r; cd[i] = c[i]
        def tour():
            a, b = rng.integers(len(pop), size=2)
            return a if (rank[a], -cd[a]) < (rank[b], -cd[b]) else b
        kids = []
        while len(kids) < N:
            pa, pb = pop[tour()], pop[tour()]
            delta = np.abs(pa - pb)                              # BLX-alpha, alpha=0.5
            child = rng.uniform(np.minimum(pa, pb) - 0.5 * delta, np.maximum(pa, pb) + 0.5 * delta)
            mut = rng.random(D) < 1.0 / D
            child = child + mut * rng.normal(0, 0.1 * (HI - LO))
            kids.append(np.clip(child, LO, HI))
        kids = np.array(kids); KF = np.array([ev(z) for z in kids])
        allp = np.vstack([pop, kids]); allF = np.vstack([F, KF]); sel = []
        for fr in nd_sort(allF):
            if len(sel) + len(fr) <= N: sel += fr
            else:
                c = crowding(allF, fr); sel += sorted(fr, key=lambda i: -c[i])[:N - len(sel)]; break
        pop, F = allp[sel], allF[sel]

def random_search(ev, rng):
    for _ in range(B): ev(rng.uniform(LO, HI))

def cmaes(ev, rng):
    import cma
    es = cma.CMAEvolutionStrategy(0.5 * np.ones(D), 0.3, {"bounds": [0, 1], "popsize": N, "seed": int(rng.integers(1e6)), "verbose": -9})
    while ev.n < B:
        xs = es.ask(); fs = []
        for x in xs:
            if ev.n >= B: fs.append(0.0); continue
            f = ev(LO + x * (HI - LO)); fs.append(-float(np.mean(f)))
        es.tell(xs, fs)

def tpe(ev, rng):
    import optuna; optuna.logging.set_verbosity(optuna.logging.ERROR)
    st = optuna.create_study(directions=["maximize", "maximize"], sampler=optuna.samplers.TPESampler(seed=int(rng.integers(1e6)), n_startup_trials=N))
    def obj(tr):
        z = np.array([tr.suggest_float(f"g{d}", LO[d], HI[d]) for d in range(D)]); f = ev(z); return float(f[0]), float(f[1])
    st.optimize(obj, n_trials=B)

def hand_tuned(ev, rng):
    ev(HAND)
    for eta in (-1.5, -1.0, -0.5):                 # small grid over learning rate and LTD/LTP ratio, no metaplasticity
        for rho in (-0.3, 0.0, 0.3):
            z = HAND.copy(); z[5] = eta; z[1] = rho; ev(z)

METHODS = {"NSGA-II": nsga2, "Random": random_search, "CMA-ES": cmaes, "TPE-MO": tpe, "Hand-tuned": hand_tuned}

# ---------------- final (held-out) evaluation ----------------
def final_eval(archive, run, gate, k=8):
    """take non-dominated archive members, re-evaluate on fresh seeds: unseen orders, and shifted-domain transfer."""
    F = np.array([[a[1], a[2]] for a in archive]); idx = pareto_idx(F)
    idx = sorted(idx, key=lambda i: F[i, 0]); 
    if len(idx) > k: idx = [idx[int(j)] for j in np.linspace(0, len(idx) - 1, k)]
    res = {"fresh": [], "shift": [], "z": []}
    for i in idx:
        z = np.array(archive[i][0]); fr, sh = [], []
        for r in range(3):
            s = 900000 + run * 100 + r; order = random_order(np.random.default_rng(s))
            fr.append(objectives(run_sequence(z, order, s, gate)))
            sh.append(objectives(run_sequence(z, order, s, gate, shifted=True)))
        res["fresh"].append(np.mean(fr, 0).tolist()); res["shift"].append(np.mean(sh, 0).tolist()); res["z"].append(z.tolist())
    res["search_front"] = F[pareto_idx(F)].tolist()
    return res

if __name__ == "__main__":
    which = sys.argv[1:] or list(METHODS)
    for m in which:
        for run in range(RUNS):
            fn = f"{OUT}/{m}_G1_run{run}.json"
            if os.path.exists(fn): continue
            t = time.time(); rng = np.random.default_rng(run + 1); ev = Evaluator(run)
            METHODS[m](ev, rng)
            fe = final_eval(ev.archive, run, "G1")
            fe["archive"] = ev.archive; fe["time"] = time.time() - t; fe["evals"] = ev.n
            json.dump(fe, open(fn, "w")); print(m, run, round(fe["time"], 1), flush=True)
    if os.environ.get("G2"):                     # gate ablation: NSGA-II with activity-based gate
        for run in range(4):
            fn = f"{OUT}/NSGA-II_G2_run{run}.json"
            if os.path.exists(fn): continue
            rng = np.random.default_rng(run + 1); ev = Evaluator(run, "G2"); nsga2(ev, rng)
            fe = final_eval(ev.archive, run, "G2"); fe["archive"] = ev.archive
            json.dump(fe, open(fn, "w")); print("G2", run, flush=True)
