"""Small statistics helpers (numpy only).

logit_cluster: logistic regression by iteratively reweighted least squares with
cluster-robust (sandwich) standard errors, clustering by town because each town
contributes several questions. wilson: confidence interval for a proportion.
"""
import numpy as np
from math import erf, sqrt


def _norm_cdf(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def logit_cluster(X, y, clusters, names, max_iter=100, tol=1e-10):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-10, None)
        H = X.T @ (X * w[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    p = 1 / (1 + np.exp(-(X @ beta)))
    bread = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    scores = X * (y - p)[:, None]
    meat = np.zeros((X.shape[1], X.shape[1]))
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    for c in uniq:
        s = scores[clusters == c].sum(0)
        meat += np.outer(s, s)
    g, n, k = len(uniq), len(y), X.shape[1]
    meat *= (g / (g - 1)) * ((n - 1) / (n - k))     # small-sample correction (as in Stata)
    cov = bread @ meat @ bread
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pvals = np.array([2 * (1 - _norm_cdf(abs(v))) for v in z])
    rows = []
    for nm, b, s, pv in zip(names, beta, se, pvals):
        rows.append({"term": nm, "coef": b, "se": s, "odds_ratio": np.exp(b),
                     "or_low": np.exp(b - 1.96 * s), "or_high": np.exp(b + 1.96 * s), "p": pv})
    return rows, beta, cov


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)
