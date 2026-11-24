"""
Hybrid EA-MOPSO
===============

Python implementation of the Hybrid Evolutionary-Algorithm Multi-Objective
Particle Swarm Optimisation described in:

    N. Islam et al., "An Improved Hybrid Multi-Objective Particle Swarm
    Optimization to Enhance Convergence and Diversity", GECCO '23.

Main components (paper Algorithms 1-5):

  * Latin Hypercube Sampling initialisation, zero initial velocity.
  * Personal best via NSPSO "combine-then-compare": the N personal bests and
    the N current particles form a 2N temporary population, which is ranked
    and the best N are kept (Algorithm 1).
  * SPEA-II strength Pareto fitness for ranking (Algorithm 2).
  * External archive that keeps only strength-0 (non-dominated) solutions and
    truncates by NSGA-II crowding distance (Algorithm 3).
  * Global best chosen by binary tournament from the archive (Section 3.4).
  * Reproduction by SBX crossover and polynomial mutation (Algorithm 4).

All objectives are MINIMISED. To maximise an objective, return its negative.

Usage
-----
    import numpy as np
    from ea_mopso import EAMOPSO

    def my_problem(X):                 # X: (n, d)  ->  F: (n, m)
        f1 = X[:, 0]
        f2 = 1 - np.sqrt(X[:, 0]) + X[:, 1:].sum(axis=1)
        return np.c_[f1, f2]

    opt = EAMOPSO(my_problem, lb=np.zeros(5), ub=np.ones(5),
                  n_particles=10, max_iter=100, seed=1)
    pareto_x, pareto_f = opt.run()
"""

import numpy as np

__all__ = [
    "EAMOPSO",
    "domination_matrix",
    "strength_fitness",
    "crowding_distance",
    "latin_hypercube",
    "sbx_crossover",
    "polynomial_mutation",
]


# =============================================================================
# Pareto utilities
# =============================================================================


def domination_matrix(F):
    """Return D with D[i, j] = True when solution i Pareto-dominates solution j."""
    F = np.asarray(F, dtype=float)
    le = (F[:, None, :] <= F[None, :, :]).all(axis=2)
    lt = (F[:, None, :] < F[None, :, :]).any(axis=2)
    return le & lt


def strength_fitness(F):
    """SPEA-II strength Pareto fitness (paper Algorithm 2).

    strength S(i) = number of solutions that i dominates
    fitness  R(j) = sum of S(i) over every solution i that dominates j

    R == 0 means non-dominated. Lower is better.
    """
    D = domination_matrix(F)
    S = D.sum(axis=1)
    return (D * S[:, None]).sum(axis=0).astype(float)


def crowding_distance(F):
    """NSGA-II crowding distance. Boundary solutions receive +inf."""
    F = np.asarray(F, dtype=float)
    n, m = F.shape
    cd = np.zeros(n)
    if n <= 2:
        cd[:] = np.inf
        return cd
    for k in range(m):
        order = np.argsort(F[:, k], kind="stable")
        fmin, fmax = F[order[0], k], F[order[-1], k]
        cd[order[0]] = cd[order[-1]] = np.inf
        if fmax > fmin:
            cd[order[1:-1]] += (F[order[2:], k] - F[order[:-2], k]) / (fmax - fmin)
    return cd


# =============================================================================
# Variation operators
# =============================================================================


def latin_hypercube(n, lb, ub, rng):
    """n points by Latin Hypercube Sampling inside [lb, ub]."""
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    d = len(lb)
    strata = np.array([rng.permutation(n) for _ in range(d)]).T
    u = (rng.random((n, d)) + strata) / n
    return lb + u * (ub - lb)


def sbx_crossover(p1, p2, lb, ub, eta, rng):
    """Simulated Binary Crossover (Deb & Agrawal), bounded version."""
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if rng.random() > 0.5 or abs(p1[i] - p2[i]) < 1e-14:
            continue
        y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
        r = rng.random()

        def beta_q(beta):
            alpha = 2.0 - beta ** -(eta + 1.0)
            if r <= 1.0 / alpha:
                return (r * alpha) ** (1.0 / (eta + 1.0))
            return (1.0 / (2.0 - r * alpha)) ** (1.0 / (eta + 1.0))

        a = 0.5 * ((y1 + y2) - beta_q(1.0 + 2.0 * (y1 - lb[i]) / (y2 - y1)) * (y2 - y1))
        b = 0.5 * ((y1 + y2) + beta_q(1.0 + 2.0 * (ub[i] - y2) / (y2 - y1)) * (y2 - y1))
        a, b = np.clip(a, lb[i], ub[i]), np.clip(b, lb[i], ub[i])
        if rng.random() < 0.5:
            a, b = b, a
        c1[i], c2[i] = a, b
    return c1, c2


def polynomial_mutation(x, lb, ub, eta, prob_var, rng):
    """Polynomial mutation (Deb), each variable mutated with probability prob_var."""
    y = x.copy()
    for i in range(len(x)):
        if rng.random() >= prob_var or ub[i] <= lb[i]:
            continue
        d1 = (y[i] - lb[i]) / (ub[i] - lb[i])
        d2 = (ub[i] - y[i]) / (ub[i] - lb[i])
        r = rng.random()
        p = 1.0 / (eta + 1.0)
        if r < 0.5:
            dq = (2.0 * r + (1.0 - 2.0 * r) * (1.0 - d1) ** (eta + 1.0)) ** p - 1.0
        else:
            dq = 1.0 - (2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (1.0 - d2) ** (eta + 1.0)) ** p
        y[i] = np.clip(y[i] + dq * (ub[i] - lb[i]), lb[i], ub[i])
    return y


# =============================================================================
# Hybrid EA-MOPSO
# =============================================================================


class EAMOPSO:
    """Hybrid EA-MOPSO optimiser.

    Parameters
    ----------
    func : callable
        func(X) -> F, X of shape (n, d), F of shape (n, m). All objectives minimised.
    lb, ub : array-like
        Lower / upper bounds of the decision variables.
    n_particles : int
        Swarm size N (paper: 10).
    max_iter : int
        Number of iterations (paper: 100).
    archive_size : int
        Maximum size of the external archive.
    w, c1, c2 : float
        Inertia weight, cognitive and social coefficients (paper: 0.5, 1.5, 1.5).
    pc : float
        Crossover probability for SBX (paper: 0.7).
    pm : float
        Probability that a particle undergoes polynomial mutation (paper: 0.7);
        inside a mutated particle each variable changes with probability 1/d.
    eta_c, eta_m : float
        Distribution indices of SBX and polynomial mutation.
    seed : int or None
        Random seed for reproducibility.
    verbose : bool
        Print progress every 10 iterations.
    """

    def __init__(self, func, lb, ub, n_particles=10, max_iter=100, archive_size=100,
                 w=0.5, c1=1.5, c2=1.5, pc=0.7, pm=0.7, eta_c=20.0, eta_m=20.0,
                 seed=None, verbose=False):
        self.func = func
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        if self.lb.shape != self.ub.shape or np.any(self.ub <= self.lb):
            raise ValueError("lb and ub must have the same length and ub > lb")
        self.n = int(n_particles)
        self.max_iter = int(max_iter)
        self.archive_size = int(archive_size)
        self.w, self.c1, self.c2 = w, c1, c2
        self.pc, self.pm = pc, pm
        self.eta_c, self.eta_m = eta_c, eta_m
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.vmax = self.ub - self.lb
        self.n_evals = 0
        self.archive_history = []   # archive size after each iteration

    # ------------------------------------------------------------------ helpers
    def _evaluate(self, X):
        F = np.asarray(self.func(X), dtype=float).reshape(len(X), -1)
        self.n_evals += len(X)
        return F

    # ------------------------------------------------- Algorithm 2 & 3: archive
    def _update_archive(self, AX, AF, X, F):
        AX = np.vstack([AX, X]) if len(AX) else X.copy()
        AF = np.vstack([AF, F]) if len(AF) else F.copy()
        _, idx = np.unique(np.round(AX, 12), axis=0, return_index=True)
        idx = np.sort(idx)
        AX, AF = AX[idx], AF[idx]
        keep = strength_fitness(AF) == 0                # only strength value 0 stays
        AX, AF = AX[keep], AF[keep]
        while len(AX) > self.archive_size:              # drop the most crowded one
            worst = int(np.argmin(crowding_distance(AF)))
            AX, AF = np.delete(AX, worst, 0), np.delete(AF, worst, 0)
        return AX, AF

    # ------------------------------------------- Algorithm 1: personal best (2N)
    def _select_pbest(self, PX, PF, X, F):
        TX, TF = np.vstack([PX, X]), np.vstack([PF, F])
        fit, cd = strength_fitness(TF), crowding_distance(TF)
        order = np.lexsort((-cd, fit))[: self.n]        # fitness asc, crowding desc
        return TX[order].copy(), TF[order].copy()

    # ------------------------------- Section 3.4: global best, binary tournament
    def _select_gbest(self, AX, AF):
        if len(AX) == 1:
            return np.repeat(AX, self.n, axis=0)
        fit, cd = strength_fitness(AF), crowding_distance(AF)
        a = self.rng.integers(0, len(AX), self.n)
        b = self.rng.integers(0, len(AX), self.n)
        a_wins = (fit[a] < fit[b]) | ((fit[a] == fit[b]) & (cd[a] >= cd[b]))
        return AX[np.where(a_wins, a, b)]

    # ------------------------------------------ Algorithm 4: SBX + poly mutation
    def _reproduce(self, X):
        d = X.shape[1]
        Y = X.copy()
        perm = self.rng.permutation(self.n)
        for k in range(0, self.n - 1, 2):
            i, j = perm[k], perm[k + 1]
            if self.rng.random() < self.pc:
                Y[i], Y[j] = sbx_crossover(X[i], X[j], self.lb, self.ub, self.eta_c, self.rng)
        for i in range(self.n):
            if self.rng.random() < self.pm:
                Y[i] = polynomial_mutation(Y[i], self.lb, self.ub, self.eta_m, 1.0 / d, self.rng)
        return Y

    # ------------------------------------------------- Algorithm 5: main loop
    def run(self):
        """Run the optimiser. Returns (pareto_x, pareto_f) from the external archive."""
        d = len(self.lb)
        X = latin_hypercube(self.n, self.lb, self.ub, self.rng)
        V = np.zeros((self.n, d))
        F = self._evaluate(X)
        PX, PF = X.copy(), F.copy()
        AX, AF = self._update_archive(np.empty((0, d)), np.empty((0, F.shape[1])), X, F)

        for it in range(self.max_iter):
            PX, PF = self._select_pbest(PX, PF, X, F)             # Alg. 1
            G = self._select_gbest(AX, AF)                        # Sec. 3.4
            r1 = self.rng.random((self.n, d))
            r2 = self.rng.random((self.n, d))
            V = self.w * V + self.c1 * r1 * (PX - X) + self.c2 * r2 * (G - X)
            V = np.clip(V, -self.vmax, self.vmax)
            Xn = X + V
            V[(Xn < self.lb) | (Xn > self.ub)] *= -0.5           # bounce off bounds
            X = np.clip(Xn, self.lb, self.ub)
            X = self._reproduce(X)                                # Alg. 4
            F = self._evaluate(X)
            AX, AF = self._update_archive(AX, AF, X, F)           # Alg. 2 & 3
            self.archive_history.append(len(AX))
            if self.verbose and (it + 1) % 10 == 0:
                print(f"iter {it + 1:4d} | evaluations {self.n_evals:6d} | archive {len(AX)}")

        return AX, AF
