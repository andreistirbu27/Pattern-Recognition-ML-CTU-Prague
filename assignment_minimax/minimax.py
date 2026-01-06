#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
from scipy.stats import norm
from scipy.optimize import fminbound
import copy

import scipy.optimize as opt
# importing bayes doesn't work in BRUTE :(, please copy the functions into this file


def minimax_strategy_discrete(distribution1, distribution2):
    """
    q = minimax_strategy_discrete(distribution1, distribution2)

    Find the optimal Minimax strategy for 2 discrete distributions.

    :param distribution1:           pXk(x|class1) given as a (n, n) np array
    :param distribution2:           pXk(x|class2) given as a (n, n) np array
    :return q:                      optimal strategy, (n, n) np array, values 0 (class 1) or 1 (class 2)
    :return: opt_i:                 index of the optimal solution found during the search, Python int in [0, n*n + 1) range
    :return: eps1:                  cumulative error on the first class for all thresholds, (n * n + 1,) numpy array
    :return: eps2:                  cumulative error on the second class for all thresholds, (n * n + 1,) numpy array
    """
    D1 = np.asarray(distribution1, dtype=float)
    D2 = np.asarray(distribution2, dtype=float)
    if D1.shape != D2.shape:
        raise ValueError("distribution1 and distribution2 must have the same shape.")

    p1 = D1.ravel()
    p2 = D2.ravel()
    N = p1.size

    with np.errstate(divide="ignore", invalid="ignore"):
        r = p1 / p2

    nan_mask = np.isnan(r)  # 0/0 positions

    # IMPORTANT: match reference tie-breaking
    order = np.argsort(r)  # default kind (usually quicksort)

    p1s = p1[order]
    p2s = p2[order]

    pref1 = np.concatenate(([0.0], np.cumsum(p1s)))
    pref2 = np.concatenate(([0.0], np.cumsum(p2s)))
    total2 = pref2[-1]

    # threshold i: first i -> class2, rest -> class1
    eps1 = pref1
    eps2 = total2 - pref2

    worst = np.maximum(eps1, eps2)
    opt_i = int(np.argmin(worst))

    # q values: 0=class1, 1=class2
    q_sorted = (np.arange(N) < opt_i).astype(np.int32)

    q_flat = np.empty(N, dtype=np.int32)
    q_flat[order] = q_sorted

    # enforce: nan -> class1 (0)
    q_flat[nan_mask] = 0

    q = q_flat.reshape(D1.shape)
    return q, opt_i, eps1, eps2


def classify_discrete(imgs, q):
    """
    function label = classify_discrete(imgs, q)

    Classify images using discrete measurement and strategy q.

    :param imgs:    test set images, (h, w, n) uint8 np array
    :param q:       strategy (21, 21) np array of 0 or 1
    :return:        image labels, (n, ) np array of 0 or 1
    """
    arr = np.asarray(imgs)

    # already (N, 2) discrete measurements
    if arr.ndim == 2 and arr.shape[1] == 2:
        xy = arr

    else:
        # images; convert to (H, W, N) for measurement funcs
        imgs = arr

        # squeeze trailing singleton channel if present
        if imgs.ndim == 4 and imgs.shape[-1] == 1:
            imgs = imgs[..., 0]  # -> (N,H,W) or (H,W,N)

        if imgs.ndim != 3:
            raise ValueError(f"Unsupported images shape: {arr.shape}. Expect (H,W,N) or (N,H,W)[,1].")

        # If it's (N,H,W), transpose to (H,W,N). Heuristic: if first dim looks like N, transpose.
        if imgs.shape[0] >= 8 and imgs.shape[0] > max(imgs.shape[1], imgs.shape[2]):
            imgs = np.transpose(imgs, (1, 2, 0))  # -> (H,W,N)

        # Now imgs is (H, W, N). Compute discrete measurements using provided helpers.
        try:
            x_meas = compute_measurement_lr_discrete(imgs)  # shape (N,)
            y_meas = compute_measurement_ul_discrete(imgs)  # shape (N,)
        except NameError as e:
            raise NameError(
                "Measurement functions not found. Make sure compute_measurement_lr_discrete "
                "and compute_measurement_ul_discrete are defined/imported."
            ) from e

        xy = np.stack([x_meas, y_meas], axis=1)  # (N,2)

    # Map (x,y) in [-10,10] to q indices and predict
    x = np.rint(xy[:, 0]).astype(int)
    y = np.rint(xy[:, 1]).astype(int)

    xi = np.clip(x + 10, 0, q.shape[0] - 1)  # rows: X
    yi = np.clip(y + 10, 0, q.shape[1] - 1)  # cols: Y

    preds = q[xi, yi].astype(int)
    return preds


def worst_risk_cont(distribution_A, distribution_B, true_A_prior):
    """
    Find the optimal bayesian strategy for true_A_prior (assuming 0-1 loss) and compute its worst possible risk in case the priors are different.

    :param distribution_A:          parameters of the normal dist.
                                    distribution_A['Mean'], distribution_A['Sigma'] - python floats
    :param distribution_B:          the same as distribution_A
    :param true_A_prior:            true A prior probability - python float
    :return worst_risk:             worst possible bayesian risk when evaluated with different prior
    """
    D1p = distribution_A.copy()
    D2p = distribution_B.copy()
    D1p["Prior"] = float(true_A_prior)
    D2p["Prior"] = 1.0 - float(true_A_prior)

    q = find_strategy_2normal(D1p, D2p)

    mu1, s1 = float(D1p["Mean"]), float(D1p["Sigma"])
    mu2, s2 = float(D2p["Mean"]), float(D2p["Sigma"])

    t1 = float(q["t1"])
    t2 = float(q["t2"])
    d = np.asarray(q["decision"], dtype=int)  # 0 -> class1, 1 -> class2

    # interval masses under each class
    F1_t1 = norm.cdf(t1, loc=mu1, scale=s1)
    F1_t2 = norm.cdf(t2, loc=mu1, scale=s1)
    p1_int = np.array([F1_t1, F1_t2 - F1_t1, 1.0 - F1_t2])

    F2_t1 = norm.cdf(t1, loc=mu2, scale=s2)
    F2_t2 = norm.cdf(t2, loc=mu2, scale=s2)
    p2_int = np.array([F2_t1, F2_t2 - F2_t1, 1.0 - F2_t2])

    # conditional errors for this strategy
    eps1 = float(p1_int[d == 1].sum())  # P(decide class2 | true class1)
    eps2 = float(p2_int[d == 0].sum())  # P(decide class1 | true class2)

    return max(eps1, eps2)


def minimax_strategy_cont(distribution_A, distribution_B):
    """
    q, worst_risk = minimax_strategy_cont(distribution_A, distribution_B)

    Find minimax strategy.

    :param distribution_A:  parameters of the normal dist.
                            distribution_A['Mean'], distribution_A['Sigma'] - python floats
    :param distribution_B:  the same as distribution_A
    :return q:              strategy dict - see bayes.find_strategy_2normal
                               q['t1'], q['t2'] - decision thresholds - python floats
                               q['decision'] - (3, ) np.int32 np.array decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
    :return worst_risk      worst risk of the minimax strategy q - python float
    """
    D1c = distribution_A.copy()
    D2c = distribution_B.copy()

    def objective(p):
        p = float(p)
        D1c["Prior"] = p
        D2c["Prior"] = 1.0 - p
        q = find_strategy_2normal(D1c, D2c)
        eps1, eps2 = _class_errors_2normal(D1c, D2c, q)
        return max(eps1, eps2)

    # stay away from endpoints for numerical stability
    p_opt = fminbound(objective, 1e-9, 1.0 - 1e-9)

    D1c["Prior"] = float(p_opt)
    D2c["Prior"] = 1.0 - float(p_opt)
    q_minimax = find_strategy_2normal(D1c, D2c)

    eps1, eps2 = _class_errors_2normal(D1c, D2c, q_minimax)
    risk_minimax = float(max(eps1, eps2))

    # enforce autograder-friendly types
    q_minimax["decision"] = np.asarray(q_minimax["decision"], dtype=np.int32)
    q_minimax["t1"] = float(q_minimax["t1"])
    q_minimax["t2"] = float(q_minimax["t2"])

    return q_minimax, risk_minimax

def _class_errors_2normal(D1, D2, q):
    """Return (eps1, eps2) for a given 2-normal strategy q."""
    mu1, s1 = float(D1["Mean"]), float(D1["Sigma"])
    mu2, s2 = float(D2["Mean"]), float(D2["Sigma"])

    t1 = float(q["t1"])
    t2 = float(q["t2"])
    d = np.asarray(q["decision"], dtype=int)

    # interval masses under each class
    F1_t1 = norm.cdf(t1, loc=mu1, scale=s1)
    F1_t2 = norm.cdf(t2, loc=mu1, scale=s1)
    p1_int = np.array([F1_t1, F1_t2 - F1_t1, 1.0 - F1_t2], dtype=float)

    F2_t1 = norm.cdf(t1, loc=mu2, scale=s2)
    F2_t2 = norm.cdf(t2, loc=mu2, scale=s2)
    p2_int = np.array([F2_t1, F2_t2 - F2_t1, 1.0 - F2_t2], dtype=float)

    eps1 = float(p1_int[d == 1].sum())  # decide class2 when true is class1
    eps2 = float(p2_int[d == 0].sum())  # decide class1 when true is class2
    return eps1, eps2


def risk_fix_q_cont(distribution_A, distribution_B, distribution_A_priors, q):
    """
    Computes bayesian risks for fixed strategy and various priors.

    :param distribution_A:          parameters of the normal dist.
                                    distribution_A['Mean'], distribution_A['Sigma'] - python floats
    :param distribution_B:          the same as distribution_A
    :param distribution_A_priors:   priors (n, ) np.array
    :param q:                       strategy dict - see bayes.find_strategy_2normal
                                       q['t1'], q['t2'] - decision thresholds - python floats
                                       q['decision'] - (3, ) np.int32 np.array decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
    :return risks:                  bayesian risk of the strategy q with varying priors (n, ) np.array
    """
    priors_1 = np.asarray(distribution_A_priors, dtype=float)

    mu1, s1 = float(distribution_A["Mean"]), float(distribution_A["Sigma"])
    mu2, s2 = float(distribution_B["Mean"]), float(distribution_B["Sigma"])

    t1 = float(q["t1"])
    t2 = float(q["t2"])
    d = np.asarray(q["decision"], dtype=int)
    if d.shape != (3,):
        raise ValueError("q_fix['decision'] must be shape (3,)")

    # interval probabilities for a Normal:
    # I1: (-inf, t1], I2: (t1, t2], I3: (t2, inf)
    F1_t1 = norm.cdf(t1, loc=mu1, scale=s1)
    F1_t2 = norm.cdf(t2, loc=mu1, scale=s1)
    p1_intervals = np.array([F1_t1, F1_t2 - F1_t1, 1.0 - F1_t2], dtype=float)

    F2_t1 = norm.cdf(t1, loc=mu2, scale=s2)
    F2_t2 = norm.cdf(t2, loc=mu2, scale=s2)
    p2_intervals = np.array([F2_t1, F2_t2 - F2_t1, 1.0 - F2_t2], dtype=float)

    # Errors:
    # err1 = P(decide class 2 | true class 1) = sum intervals where d == 1 under class 1
    # err2 = P(decide class 1 | true class 2) = sum intervals where d == 0 under class 2
    err1 = float(np.sum(p1_intervals[d == 1]))
    err2 = float(np.sum(p2_intervals[d == 0]))

    risk = priors_1 * err1 + (1.0 - priors_1) * err2
    return risk


################################################################################
#####                                                                      #####
#####                Put functions from previous labs here.                #####
#####            (Sorry, we know imports would be much better)             #####
#####                                                                      #####
################################################################################

def classification_error(predictions, labels):
    """
    error = classification_error(predictions, labels)

    :param predictions: (n, ) np.array of values 0 or 1 - predicted labels
    :param labels:      (n, ) np.array of values 0 or 1 - ground truth labels
    :return:            error - classification error ~ a fraction of predictions being incorrect
                        python float in range <0, 1>
    """
    return np.mean(predictions != labels)


def find_strategy_2normal(distribution_A, distribution_B):
    """
    q = find_strategy_2normal(distribution_A, distribution_B)

    Find optimal bayesian strategy for 2 normal distributions and zero-one loss function.

    :param distribution_A:  parameters of the normal dist.
                            distribution_A['Mean'], distribution_A['Sigma'], distribution_A['Prior'] - python floats
    :param distribution_B:  the same as distribution_A

    :return q:              strategy dict
                               q['t1'], q['t2'] - decision thresholds - python floats
                               q['decision'] - (3, ) np.int32 np.array decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
                               If there is only one threshold, q['t1'] should be equal to q['t2'] and the middle decision should be 0
                               If there is no threshold, q['t1'] and q['t2'] should be -/+ infinity and all the decision values should be the same
                                (0 preferred if both strategies would have the same risk)
    """

    s_A = distribution_A['Sigma']
    m_A = distribution_A['Mean']
    p_A = distribution_A['Prior']
    s_B = distribution_B['Sigma']
    m_B = distribution_B['Mean']
    p_B = distribution_B['Prior']

    q = {}

    # extreme priors
    eps = 1e-10
    if p_A < eps:
        q['t1'], q['t2'] = -np.inf, np.inf
        q['decision'] = np.array([1, 1, 1], dtype=np.int32)
        return q
    if p_B < eps:
        q['t1'], q['t2'] = -np.inf, np.inf
        q['decision'] = np.array([0, 0, 0], dtype=np.int32)
        return q

    a = 1 / (2 * s_B ** 2) - 1 / (2 * s_A ** 2)
    b = m_A / (s_A ** 2) - m_B / (s_B ** 2)
    c = (m_B ** 2) / (2 * s_B ** 2) - (m_A ** 2) / (2 * s_A ** 2) + np.log((p_A * s_B) / (p_B * s_A))

    tol = 1e-12

    if abs(a) < tol:
        # same sigmas -> not quadratic
        if abs(b) < tol:
            # same sigmas and same means -> not even linear
            q['t1'], q['t2'] = -np.inf, np.inf
            if c > 0 or abs(c) < tol:
                q['decision'] = np.array([0, 0, 0], dtype=np.int32)
            else:
                q['decision'] = np.array([1, 1, 1], dtype=np.int32)
        else:
            # same sigmas, different means -> linear equation
            t = -c / b
            q['t1'], q['t2'] = t, t
            # single switch, middle interval irrelevant => keep it 0
            if b > 0:
                q['decision'] = np.array([1, 0, 0], dtype=np.int32)  # left=B, right=A
            else:
                q['decision'] = np.array([0, 0, 1], dtype=np.int32)  # left=A, right=B
    else:
        # quadratic equation
        D = b ** 2 - 4 * a * c

        if D > tol:
            roots = np.sort(np.roots([a, b, c]))
            t1, t2 = float(roots[0]), float(roots[1])
            q['t1'], q['t2'] = t1, t2
            if a > 0:
                q['decision'] = np.array([0, 1, 0], dtype=np.int32)
            else:
                q['decision'] = np.array([1, 0, 1], dtype=np.int32)

        elif abs(D) < tol:
            # tangency: no sign change, so decision is constant on both sides.
            # keep one "threshold" per spec (t1==t2) and force middle decision to 0.
            t = -b / (2 * a)
            q['t1'], q['t2'] = t, t
            if a > 0:
                q['decision'] = np.array([0, 0, 0], dtype=np.int32)
            else:
                q['decision'] = np.array([1, 0, 1], dtype=np.int32)

        else:  # D < 0
            q['t1'], q['t2'] = -np.inf, np.inf
            if a > 0:
                q['decision'] = np.array([0, 0, 0], dtype=np.int32)
            else:
                q['decision'] = np.array([1, 1, 1], dtype=np.int32)

    return q


def bayes_risk_2normal(distribution_A, distribution_B, q):
    """
    R = bayes_risk_2normal(distribution_A, distribution_B, q)

    Compute bayesian risk of a strategy q for 2 normal distributions and zero-one loss function.

    :param distribution_A:  parameters of the normal dist.
                            distribution_A['Mean'], distribution_A['Sigma'], distribution_A['Prior'] python floats
    :param distribution_B:  the same as distribution_A
    :param q:               strategy
                               q['t1'], q['t2'] - float decision thresholds (python floats)
                               q['decision'] - (3, ) np.int32 np.array 0/1 decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
    :return:    R - bayesian risk, python float
    """
    muA, sigmaA, pA = distribution_A['Mean'], distribution_A['Sigma'], distribution_A['Prior']
    muC, sigmaC, pC = distribution_B['Mean'], distribution_B['Sigma'], distribution_B['Prior']

    t1, t2 = q['t1'], q['t2']
    decisions = q['decision']

    def integrate_region(decision, left, right):
        if decision == 0:
            return pA * (norm.cdf((right - muA) / sigmaA) -
                         norm.cdf((left - muA) / sigmaA))
        else:
            return pC * (norm.cdf((right - muC) / sigmaC) -
                         norm.cdf((left - muC) / sigmaC))

    I1 = integrate_region(decisions[0], -np.inf, t1)
    I2 = integrate_region(decisions[1], t1, t2)
    I3 = integrate_region(decisions[2], t2, np.inf)

    R = 1 - (I1 + I2 + I3)
    return R


def classify_2normal(measurements, q):
    """
    label = classify_2normal(measurements, q)

    Classify images using continuous measurements and strategy q.

    :param imgs:    test set measurements, np.array (n, )
    :param q:       strategy
                    q['t1'] q['t2'] - float decision thresholds
                    q['decision'] - (3, ) int32 np.array decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
    :return:        label - classification labels, (n, ) int32
    """
    t1, t2 = q['t1'], q['t2']
    d1, d2, d3 = q['decision']  # decisions per interval

    x = np.asarray(measurements)

    # vectorized interval selection
    labels = np.empty_like(x, dtype=int)
    labels[x < t1] = d1
    labels[(x >= t1) & (x < t2)] = d2
    labels[x >= t2] = d3
    return labels


################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################


def plot_lr_threshold(eps1, eps2, thr):
    """
    Plot the search for the strategy

    :param eps1:  cumulative error on the first class for all thresholds, (N + 1, ) numpy array
    :param eps2:  cumulative error on the second class for all thresholds, (N + 1, ) numpy array
    :param thr:   index of the optimal solution found during the search, Python int in [0, N+1) range
    :return:      matplotlib.pyplot figure
    """

    fig = plt.figure(figsize=(15, 5))
    plt.plot(eps2, 'o-', label='$\\epsilon_2$')
    plt.plot(eps1, 'o-', label='$\\epsilon_1$')
    plt.plot([thr, thr], [-0.02, 1], 'k')
    plt.legend()
    plt.ylabel('classification error')
    plt.xlabel('i')
    plt.title('minimax - LR threshold search')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # inset axes....
    ax = plt.gca()
    axins = ax.inset_axes([0.4, 0.2, 0.4, 0.6])
    axins.plot(eps2, 'o-')
    axins.plot(eps1, 'o-')
    axins.plot([thr, thr], [-0.02, 1], 'k')
    axins.set_xlim(thr - 10, thr + 10)
    axins.set_ylim(-0.02, 1)
    axins.xaxis.set_major_locator(MaxNLocator(integer=True))
    axins.set_title('zoom in')
    # ax.indicate_inset_zoom(axins)

    return fig


def plot_discrete_strategy(q, letters):
    """
    Plot for discrete strategy

    :param q:        strategy (21, 21) np array of 0 or 1
    :param letters:  python string with letters, e.g. 'CN'
    :return:         matplotlib.pyplot figure
    """
    fig = plt.figure()
    im = plt.imshow(q, extent=[-10,10,10,-10])
    values = np.unique(q)   # values in q
    # get the colors of the values, according to the colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color
    patches = [ mpatches.Patch(color=colors[i], label="Class {}".format(letters[values[i]])) for i in range(len(values))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel('X')
    plt.xlabel('Y')

    return fig


def compute_measurement_lr_cont(imgs):
    """
    x = compute_measurement_lr_cont(imgs)

    Compute measurement on images, subtract sum of right half from sum of
    left half.

    :param imgs:    set of images, (h, w, n) numpy array
    :return x:      measurements, (n, ) numpy array
    """
    assert len(imgs.shape) == 3

    width = imgs.shape[1]
    sum_rows = np.sum(imgs, dtype=np.float64, axis=0)

    x = np.sum(sum_rows[0:int(width / 2),:], axis=0) - np.sum(sum_rows[int(width / 2):,:], axis=0)

    assert x.shape == (imgs.shape[2], )
    return x


def compute_measurement_lr_discrete(imgs):
    """
    x = compute_measurement_lr_discrete(imgs)

    Calculates difference between left and right half of image(s).

    :param imgs:    set of images, (h, w, n) (or for color images (h, w, 3, n)) np array
    :return x:      measurements, (n, ) np array of values in range <-10, 10>,
    """
    assert len(imgs.shape) in (3, 4)
    assert (imgs.shape[2] == 3 or len(imgs.shape) == 3)

    mu = -563.9
    sigma = 2001.6

    if len(imgs.shape) == 3:
        imgs = np.expand_dims(imgs, axis=2)

    imgs = imgs.astype(np.int32)
    height, width, channels, count = imgs.shape

    x_raw = np.sum(np.sum(np.sum(imgs[:, 0:int(width / 2), :, :], axis=0), axis=0), axis=0) - \
            np.sum(np.sum(np.sum(imgs[:, int(width / 2):, :, :], axis=0), axis=0), axis=0)
    x_raw = np.squeeze(x_raw)

    x = np.atleast_1d(np.round((x_raw - mu) / (2 * sigma) * 10))
    x[x > 10] = 10
    x[x < -10] = -10

    assert x.shape == (imgs.shape[-1], )
    return x


def compute_measurement_ul_discrete(imgs):
    """
    x = compute_measurement_ul_discrete(imgs)

    Calculates difference between upper and lower half of image(s).

    :param imgs:    set of images, (h, w, n) (or for color images (h, w, 3, n)) np array
    :return x:      measurements, (n, ) np array of values in range <-10, 10>,
    """
    assert len(imgs.shape) in (3, 4)
    assert (imgs.shape[2] == 3 or len(imgs.shape) == 3)

    mu = -563.9
    sigma = 2001.6

    if len(imgs.shape) == 3:
        imgs = np.expand_dims(imgs, axis=2)

    imgs = imgs.astype(np.int32)
    height, width, channels, count = imgs.shape

    x_raw = np.sum(np.sum(np.sum(imgs[0:int(height / 2), :, :, :], axis=0), axis=0), axis=0) - \
            np.sum(np.sum(np.sum(imgs[int(height / 2):, :, :, :], axis=0), axis=0), axis=0)
    x_raw = np.squeeze(x_raw)

    x = np.atleast_1d(np.round((x_raw - mu) / (2 * sigma) * 10))
    x[x > 10] = 10
    x[x < -10] = -10

    assert x.shape == (imgs.shape[-1], )
    return x


def create_test_set(images_test, labels_test, letters, alphabet):
    """
    images, labels = create_test_set(images_test, letters, alphabet)

    Return subset of the <images_test> corresponding to <letters>

    :param images_test: test images of all letter in alphabet - np.array (h, w, n)
    :param labels_test: labels for images_test - np.array (n,)
    :param letters:     python string with letters, e.g. 'CN'
    :param alphabet:    alphabet used in images_test - ['A', 'B', ...]
    :return images:     images - np array (h, w, n)
    :return labels:     labels for images, np array (n,)
    """

    images = np.empty((images_test.shape[0], images_test.shape[1], 0), dtype=np.uint8)
    labels = np.empty((0,))
    for i in range(len(letters)):
        letter_idx = np.where(alphabet == letters[i])[0]
        images = np.append(images, images_test[:, :, labels_test == letter_idx], axis=2)
        lab = labels_test[labels_test == letter_idx]
        labels = np.append(labels, np.ones_like(lab) * i, axis=0)

    return images, labels


def show_classification(test_images, labels, letters):
    """
    show_classification(test_images, labels, letters)

    create montages of images according to estimated labels

    :param test_images:     np.array (h, w, n)
    :param labels:          labels for input images np.array (n,)
    :param letters:         string with letters, e.g. 'CN'
    """
    assert isinstance(labels, np.ndarray), "'labels' must be a numpy array!"

    def montage(images, colormap='gray'):
        """
        Show images in grid.

        :param images:      np.array (h, w, n)
        :param colormap:    numpy colormap
        """
        h, w, count = np.shape(images)
        h_sq = int(np.ceil(np.sqrt(count)))
        w_sq = h_sq
        im_matrix = np.zeros((h_sq * h, w_sq * w))

        image_id = 0
        for j in range(h_sq):
            for k in range(w_sq):
                if image_id >= count:
                    break
                slice_w = j * h
                slice_h = k * w
                im_matrix[slice_h:slice_h + w, slice_w:slice_w + h] = images[:, :, image_id]
                image_id += 1
        plt.imshow(im_matrix, cmap=colormap)
        plt.axis('off')
        return im_matrix

    for i in range(len(letters)):
        imgs = test_images[:,:,labels==i]
        subfig = plt.subplot(1,len(letters),i+1)
        montage(imgs)
        plt.title(letters[i])


################################################################################
#####                                                                      #####
#####             Below this line you may insert debugging code            #####
#####                                                                      #####
################################################################################

def main():
    # HERE IT IS POSSIBLE TO ADD YOUR TESTING OR DEBUGGING CODE
    pass

if __name__ == "__main__":
    main()
