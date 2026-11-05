"""Figure generation (matplotlib, static PNGs for the paper).

Colour roles follow a fixed, colour-blind-validated categorical order
(blue, orange, aqua) with a neutral grey for the non-learned baseline.
Classes: benign = blue, attack = orange - used consistently in every figure.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
MODEL_COLORS = {"Rule (>=5 failures)": GREY, "Logistic Regression": BLUE,
                "Random Forest": ORANGE, "HistGradientBoosting": AQUA}
CLASS_COLORS = {"benign": BLUE, "attack": ORANGE}
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2,
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def roc_and_pr(y_true, score_dict, roc_path, pr_path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], ls=":", color=GREY, lw=1)
    for name, s in score_dict.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        from sklearn.metrics import roc_auc_score
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], ls="--" if name.startswith("Rule") else "-",
                label=f"{name} (AUC {roc_auc_score(y_true, s):.3f})")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate (recall)",
           title="ROC - held-out test sessions", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right", fontsize=8.5)
    # zoomed inset on the operating region that matters for a SOC (low FPR)
    ins = ax.inset_axes([0.42, 0.33, 0.52, 0.38])
    for name, s in score_dict.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        ins.plot(fpr, tpr, color=MODEL_COLORS[name], lw=1.5,
                 ls="--" if name.startswith("Rule") else "-")
    ins.set(xlim=(0, 0.05), ylim=(0.8, 1.005), title="zoom: FPR <= 5%")
    ins.title.set_fontsize(8); ins.tick_params(labelsize=7)
    _save(fig, roc_path)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    base = float(np.mean(y_true))
    ax.axhline(base, ls=":", color=GREY, lw=1)
    ax.annotate(f"prevalence {base:.3f}", (0.02, base + 0.015), color=INK2, fontsize=8)
    from sklearn.metrics import average_precision_score
    for name, s in score_dict.items():
        p, r, _ = precision_recall_curve(y_true, s)
        ax.plot(r, p, color=MODEL_COLORS[name], ls="--" if name.startswith("Rule") else "-",
                label=f"{name} (AP {average_precision_score(y_true, s):.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall - held-out test sessions",
           xlim=(0, 1.01), ylim=(0, 1.02))
    ax.legend(loc="lower left", fontsize=8.5)
    _save(fig, pr_path)


def confusion_matrices(cms: dict, path):
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.2))
    for ax, (name, cm) in zip(axes, cms.items()):
        cm = np.asarray(cm)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({norm[i, j]:.1%})", ha="center", va="center",
                        fontsize=9, color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], ["benign", "attack"]); ax.set_yticks([0, 1], ["benign", "attack"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(name, fontsize=10); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("Confusion matrices (test set, threshold 0.5; row-normalised shading)",
                 fontsize=11, fontweight="bold")
    _save(fig, path)


def feature_importance(perm_df, mdi, lr_coef, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    m = mdi.sort_values()
    axes[0].barh(m.index, m.values, color=ORANGE, height=0.6)
    axes[0].set(title="Random Forest - impurity importance (MDI)", xlabel="Share of total impurity decrease")
    d = perm_df.reindex(m.index)
    axes[1].barh(d.index, d["mean"], xerr=d["std"], color=ORANGE, height=0.6,
                 error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    axes[1].set(title="Random Forest - permutation importance", xlabel="Mean drop in average precision (test)")
    c = lr_coef.reindex(m.index)
    axes[2].barh(c.index, c.values, color=[BLUE if v >= 0 else GREY for v in c.values], height=0.6)
    axes[2].axvline(0, color=INK2, lw=1)
    axes[2].set(title="Logistic Regression - standardised coef.", xlabel="Coefficient (+ pushes toward attack)")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    _save(fig, path)


def failure_ratio_by_class(sessions, path):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    bins = np.linspace(0, 1, 11)
    w = (bins[1] - bins[0]) * 0.42
    centers = (bins[:-1] + bins[1:]) / 2
    for k, (lab, name) in enumerate([(0, "benign"), (1, "attack")]):
        v = sessions.loc[sessions["is_attack"] == lab, "failure_ratio"].clip(0, 1)
        h, _ = np.histogram(v, bins=bins)
        frac = h / h.sum()
        ax.bar(centers + (k - 0.5) * w * 1.05, frac, width=w, color=CLASS_COLORS[name],
               label=f"{name} (n={len(v):,}, median {v.median():.2f})")
    ax.set(xlabel="Session failure ratio (failed / total attempts)", ylabel="Share of sessions in class",
           title="Failure-ratio distribution by class (classic attacks)", xticks=bins)
    ax.legend(loc="upper center")
    _save(fig, path)


def evasion_bars(table, path, title):
    """Grouped bars: detection rate per attack type (rows) for each model (cols)."""
    fig, ax = plt.subplots(figsize=(7.6, 4))
    types = list(table.index)
    models = list(table.columns)
    x = np.arange(len(types))
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        ax.bar(x + (k - (len(models) - 1) / 2) * w, table[m], width=w * 0.92,
               color=MODEL_COLORS.get(m, BLUE), label=m)
        for xi, v in zip(x, table[m]):                       # label near-zero bars explicitly
            if v < 0.3:
                ax.text(xi + (k - (len(models) - 1) / 2) * w, v + 0.015, f"{v:.3f}",
                        ha="center", fontsize=7, color=INK2, rotation=90)
    ax.set_xticks(x, [t.replace("_", " ") for t in types])
    ax.set(ylim=(0, 1.05), ylabel="Share of sessions flagged", title=title)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    _save(fig, path)


def window_ablation(table, path):
    """Detection rate per attack type for RF under two aggregation windows."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    types = list(table.index)
    x = np.arange(len(types))
    cols = list(table.columns)
    colors = [BLUE, ORANGE]
    for k, c in enumerate(cols):
        ax.bar(x + (k - 0.5) * 0.38, table[c], width=0.36, color=colors[k], label=c)
        for xi, v in zip(x, table[c]):
            if np.isnan(v):
                continue
            ax.text(xi + (k - 0.5) * 0.38, v + 0.015, f"{v:.3f}" if v < 0.1 else f"{v:.2f}",
                    ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [t.replace("_", "\n") for t in types], fontsize=8.5)
    ax.axvline(3.5, color=GRID, lw=1.2)
    ax.text(1.5, 1.07, "benign actors (flag rate = FPR)", ha="center", fontsize=8.5, color=INK2)
    ax.text(5.5, 1.07, "attack types (flag rate = recall)", ha="center", fontsize=8.5, color=INK2)
    ax.set(ylim=(0, 1.13), ylabel="Share of sessions flagged\n(benign = false-positive rate)",
           title="Random Forest trained on all attack types: effect of aggregation window")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8.5, loc="center left")
    _save(fig, path)
