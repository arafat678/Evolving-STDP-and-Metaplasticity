"""Spiking network with STDP + metaplasticity + homeostasis, continual learning on Split-Digits.
Variant A (fully local). Task-incremental scenario with per-task Diehl&Cook-style neuron labelling.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ---------------- data ----------------
_d = load_digits()
X_ALL = (_d.data / 16.0).astype(np.float32)
Y_ALL = _d.target
Xtr, Xte, Ytr, Yte = train_test_split(X_ALL, Y_ALL, test_size=0.4, random_state=0, stratify=Y_ALL)

def shift_domain(X, rng):
    """Domain-shifted variant (transfer test): random 1-px shift + pixel dropout + noise."""
    out = np.empty_like(X)
    for n, x in enumerate(X):
        im = x.reshape(8, 8)
        im = np.roll(im, rng.integers(-1, 2), axis=0)
        im = np.roll(im, rng.integers(-1, 2), axis=1)
        im = im * (rng.random((8, 8)) > 0.2) + 0.1 * rng.random((8, 8))
        out[n] = np.clip(im, 0, 1).ravel()
    return out

# ---------------- genome ----------------
# gene: (name, low, high, scale)   (log10 for 'log')
GENES = [
    ("A_plus", -3.0, -0.5, "log"),
    ("rho",    -0.7,  0.7, "log"),     # A_minus / A_plus
    ("tau_p",   0.5,  1.6, "log"),     # trace time constants (time steps)
    ("tau_m_",  0.5,  1.6, "log"),
    ("meta",   -2.0,  1.5, "log"),     # m (G1) or 1/c0 (G2); below -1.5 => 0 (plain STDP)
    ("eta",    -1.5,  0.5, "log"),
    ("theta0",  0.5,  5.0, "lin"),
    ("w_inh",  -1.0,  1.0, "log"),
    ("gamma",  -2.5, -0.5, "log"),
    ("tau_th",  1.0,  3.0, "log"),
]
D = len(GENES)
LO = np.array([g[1] for g in GENES]); HI = np.array([g[2] for g in GENES])

def decode(z):
    p = {}
    for (name, lo, hi, sc), v in zip(GENES, z):
        v = float(np.clip(v, lo, hi))
        p[name] = 10 ** v if sc == "log" else v
    if z[4] < -1.5: p["meta"] = 0.0
    return p

HAND = np.array([-1.5, np.log10(1.0), 1.2, 1.2, -2.0, -1.0, 2.0, 0.0, -1.5, 2.0])  # Diehl&Cook-like, no metaplasticity
HAND[1] = 0.0

# ---------------- network ----------------
N_IN, N_H, K, PIN = 64, 24, 20, 0.25   # PIN = r_max*dt
W_SUM, W_MAX, BETA = 8.0, 1.0, 0.85

def run_poisson(X, rng):
    return (rng.random((len(X), K, N_IN)) < X[:, None, :] * PIN).astype(np.float32)

def init_net(rng):
    W = rng.random((N_H, N_IN)).astype(np.float32)
    W *= W_SUM / W.sum(1, keepdims=True)
    return W

def train_sample(W, th, C, spikes, p, gate):
    """one sample, STDP with traces, inhibition, adaptive threshold, gate, normalisation."""
    U = np.zeros(N_H, np.float32); S = np.zeros(N_H, np.float32)
    xt = np.zeros(N_IN, np.float32); yt = np.zeros(N_H, np.float32)
    dW = np.zeros_like(W)
    ap, am = p["A_plus"], p["A_plus"] * p["rho"]
    dx, dy = np.exp(-1 / p["tau_p"]), np.exp(-1 / p["tau_m_"])
    decay = 1.0 / p["tau_th"]**1 if False else 1.0 / (10 ** 2)  # slow threshold relaxation, fixed time-scale
    tau_th = p["tau_th"]
    for k in range(K):
        s_in = spikes[k]
        I = W @ s_in
        U = BETA * U + I - th * S - p["w_inh"] * (S.sum() - S)
        S = (U >= th).astype(np.float32)
        xt = dx * xt + s_in; yt = dy * yt + S
        if S.any() or s_in.any():
            dW += ap * np.outer(S, xt) - am * np.outer(yt, s_in)
        th = th + p["gamma"] * S - (th - p["theta0"]) / (tau_th * 20)
    th_new = th
    # gate
    if p["meta"] > 0:
        if gate == "G1":
            g = np.where(dW < 0, 1 - np.tanh(p["meta"] * W) ** 2, 1.0)
        else:  # G2 activity-based consolidation
            C *= (1 - 0.01); C += 0.5 * np.abs(dW)
            g = np.exp(-C * p["meta"] * 5)
        dW = dW * g
    W += p["eta"] * dW
    np.clip(W, 0, W_MAX, out=W)
    W *= W_SUM / (W.sum(1, keepdims=True) + 1e-9)   # normalisation
    np.clip(W, 0, W_MAX, out=W)
    return th_new

def respond(W, th, spikes_b):
    """batched inference (no learning): returns spike counts (n_samples, N_H)."""
    B = spikes_b.shape[0]
    U = np.zeros((B, N_H), np.float32); S = np.zeros((B, N_H), np.float32); n = np.zeros((B, N_H), np.float32)
    for k in range(K):
        I = spikes_b[:, k, :] @ W.T
        U = BETA * U + I - th * S - INH_EVAL * (S.sum(1, keepdims=True) - S)
        S = (U >= th).astype(np.float32); n += S
    return n

INH_EVAL = 0.0

def run_sequence(z, order, seed, gate="G1", n_train=70, shifted=False, return_extra=False):
    """order: list of T class pairs. returns accuracy matrix a[t,i]."""
    global INH_EVAL
    p = decode(z); INH_EVAL = p["w_inh"]
    rng = np.random.default_rng(seed)
    W = init_net(rng); th = np.full(N_H, p["theta0"], np.float32); C = np.zeros_like(W)
    T = len(order)
    a = np.zeros((T, T)); labmaps = []
    Xte_ = shift_domain(Xte, np.random.default_rng(seed + 999)) if shifted else Xte
    for t, cls in enumerate(order):
        idx = np.where(np.isin(Ytr, cls))[0]
        idx = rng.permutation(idx)[:n_train]
        Xt = shift_domain(Xtr[idx], rng) if shifted else Xtr[idx]
        sp = run_poisson(Xt, rng)
        for s in sp:
            th = train_sample(W, th, C, s, p, gate)
        # label neurons on this task's training data (local response averaging)
        n = respond(W, th, sp)
        Rr = np.stack([n[Ytr[idx] == c].mean(0) if (Ytr[idx] == c).any() else np.zeros(N_H) for c in cls], 1)
        labmaps.append(Rr.argmax(1))       # neuron j -> index into cls
        # evaluate on all tasks seen so far
        for i in range(t + 1):
            ci = order[i]; ti = np.where(np.isin(Yte, ci))[0]
            ns = respond(W, th, run_poisson(Xte_[ti], np.random.default_rng(seed + 7 * i)))
            lm = labmaps[i]
            score = np.stack([ns[:, lm == c].mean(1) if (lm == c).any() else np.zeros(len(ti)) for c in range(len(ci))], 1)
            pred = np.array(ci)[score.argmax(1)]
            a[t, i] = (pred == Yte[ti]).mean()
    if return_extra:
        return a, dict(sat=float((W > 0.9 * W_MAX).mean()), W=W)
    return a

def random_order(rng):
    perm = rng.permutation(10)
    return [tuple(perm[2 * i:2 * i + 2]) for i in range(5)]

def objectives(a):
    T = a.shape[0]
    f_pl = np.mean(np.diag(a)); f_st = np.mean(a[T - 1, :T - 1])
    return f_pl, f_st

def cl_metrics(a):
    T = a.shape[0]
    acc = a[T - 1].mean()
    bwt = np.mean([a[T - 1, i] - a[i, i] for i in range(T - 1)])
    fgt = np.mean([max(a[l, i] for l in range(i, T - 1)) - a[T - 1, i] for i in range(T - 1)])
    return acc, bwt, fgt

def fitness(z, seeds, gate="G1"):
    fs = []
    for s in seeds:
        order = random_order(np.random.default_rng(s))
        fs.append(objectives(run_sequence(z, order, s, gate)))
    return np.mean(fs, 0)
