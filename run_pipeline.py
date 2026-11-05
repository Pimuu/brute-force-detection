#!/usr/bin/env python3
"""End-to-end reproducible pipeline: data -> features -> models -> results/figures.

Usage:  python run_pipeline.py            (≈30-60 s on a laptop CPU)

Experiments
  E1  Main comparison on classic attacks (brute force + credential stuffing),
      stratified 70/30 split + 5-fold stratified cross-validation.
  E2  Evasion: E1 models applied, unchanged, to held-out low-and-slow and
      distributed (botnet) attacks they never saw in training.
  E3  Mitigation ablation: 30-min sessions vs. 24-h per-IP windows, Random Forest
      trained on *all* attack types.
  E4  Robustness: E1 repeated on 5 independently generated datasets.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, train_test_split

from src import config as C
from src import plots
from src.data_generation import generate_events
from src.evaluate import binary_metrics, scores
from src.features import build_session_table
from src.models import make_models

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def split(sessions, seed):
    X, y = sessions[C.FEATURES], sessions["is_attack"].to_numpy()
    idx_tr, idx_te = train_test_split(np.arange(len(sessions)), test_size=C.TEST_SIZE,
                                      stratify=y, random_state=seed)
    return X.iloc[idx_tr], X.iloc[idx_te], y[idx_tr], y[idx_te], idx_tr, idx_te


def main():
    C.DATA_DIR.mkdir(exist_ok=True); C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    events = generate_events(C.SEED)
    events.to_csv(C.DATA_DIR / "login_events.csv.gz", index=False)
    log(f"generated {len(events):,} login events")

    sessions = build_session_table(events)
    sessions.to_csv(C.DATA_DIR / "sessions.csv.gz", index=False)
    log(f"extracted {len(sessions):,} sessions")

    summary = events.pivot_table(index="actor", columns="window", values="ip", aggfunc="size",
                                 fill_value=0).add_prefix("events_")
    summary["eval_sessions"] = sessions.groupby("kind").size()
    summary["is_attack"] = summary.index.isin([*C.CLASSIC_ATTACKS, *C.EVASIVE_ATTACKS]).astype(int)
    summary = summary.fillna(0).astype(int).sort_values(["is_attack", "eval_sessions"], ascending=[True, False])
    summary.loc["TOTAL"] = summary.sum()
    summary.loc["TOTAL", "is_attack"] = -1   # not applicable
    summary.to_csv(C.RESULTS_DIR / "dataset_summary.csv")

    classic = sessions[sessions["attack_type"].isin(["benign", *C.CLASSIC_ATTACKS])].reset_index(drop=True)
    evasive = sessions[sessions["attack_type"].isin(C.EVASIVE_ATTACKS)].reset_index(drop=True)

    # ------------------------------------------------------------------ E1
    Xtr, Xte, ytr, yte, _, idx_te = split(classic, C.SEED)
    test_rows = classic.iloc[idx_te].reset_index(drop=True)
    fitted, test_scores, rows, cms, per_type = {}, {}, [], {}, {}
    for name, model in make_models(C.SEED).items():
        m = clone(model).fit(Xtr, ytr)
        s = scores(m, Xte)
        fitted[name], test_scores[name] = m, s
        met = binary_metrics(yte, s)
        rows.append({"model": name, **met})
        cms[name] = [[met["tn"], met["fp"]], [met["fn"], met["tp"]]]
        per_type[name] = pd.Series(s >= C.THRESHOLD).groupby(test_rows["kind"]).mean()
    metrics = pd.DataFrame(rows)
    metrics.to_csv(C.RESULTS_DIR / "metrics.csv", index=False, float_format="%.4f")
    pt = pd.DataFrame(per_type)
    pt.to_csv(C.RESULTS_DIR / "test_flag_rate_by_kind.csv", float_format="%.4f")
    log("E1 test metrics:\n" + metrics.round(4).to_string(index=False))

    # error analysis: what do the Random Forest false positives / negatives look like?
    rf_pred = test_scores["Random Forest"] >= C.THRESHOLD
    err = test_rows.assign(score=test_scores["Random Forest"])
    errs = err[(rf_pred & (yte == 0)) | (~rf_pred & (yte == 1))]
    errs[["kind", "ip", "start", *C.FEATURES, "score"]].to_csv(
        C.RESULTS_DIR / "rf_errors.csv", index=False, float_format="%.4f")

    # 5-fold stratified CV on the classic set
    cv_rows = []
    skf = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.SEED)
    X_all, y_all = classic[C.FEATURES], classic["is_attack"].to_numpy()
    for fold, (tr, te) in enumerate(skf.split(X_all, y_all)):
        for name, model in make_models(C.SEED).items():
            m = clone(model).fit(X_all.iloc[tr], y_all[tr])
            cv_rows.append({"model": name, "fold": fold,
                            **binary_metrics(y_all[te], scores(m, X_all.iloc[te]))})
    cv = pd.DataFrame(cv_rows)
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr"]
    cv_summary = cv.groupby("model", sort=False)[cols].agg(["mean", "std"])
    cv_summary.columns = [f"{a}_{b}" for a, b in cv_summary.columns]
    cv_summary.to_csv(C.RESULTS_DIR / "cv_metrics.csv", float_format="%.4f")
    log("E1 5-fold CV (mean):\n" + cv.groupby("model", sort=False)[cols].mean().round(4).to_string())

    # feature importance
    rf = fitted["Random Forest"]
    pi = permutation_importance(rf, Xte, yte, scoring="average_precision", n_repeats=10,
                                random_state=C.SEED, n_jobs=-1)
    perm = pd.DataFrame({"mean": pi.importances_mean, "std": pi.importances_std}, index=C.FEATURES)
    lr = fitted["Logistic Regression"]
    order = C.COUNT_FEATURES + [f for f in C.FEATURES if f not in C.COUNT_FEATURES]
    coef = pd.Series(lr.named_steps["clf"].coef_[0], index=order).reindex(C.FEATURES)
    imp = pd.DataFrame({"rf_permutation_mean": perm["mean"], "rf_permutation_std": perm["std"],
                        "rf_impurity": rf.feature_importances_, "lr_std_coef": coef})
    imp.to_csv(C.RESULTS_DIR / "feature_importance.csv", float_format="%.4f")

    # ------------------------------------------------------------------ E2
    ev_rows = []
    for name, m in fitted.items():
        s = scores(m, evasive[C.FEATURES])
        flagged = pd.Series(s >= C.THRESHOLD)
        for t in C.EVASIVE_ATTACKS:
            mask = (evasive["attack_type"] == t).to_numpy()
            camp = flagged[mask].groupby(evasive.loc[mask, "campaign_id"].to_numpy()).any()
            ev_rows.append({"model": name, "attack_type": t, "sessions": int(mask.sum()),
                            "session_detection_rate": flagged[mask].mean(),
                            "campaigns": int(camp.size), "campaign_detection_rate": camp.mean()})
        for t in C.CLASSIC_ATTACKS:
            ev_rows.append({"model": name, "attack_type": t, "sessions": int((test_rows.attack_type == t).sum()),
                            "session_detection_rate": per_type[name].get(t, np.nan),
                            "campaigns": np.nan, "campaign_detection_rate": np.nan})
    evasion = pd.DataFrame(ev_rows)
    evasion.to_csv(C.RESULTS_DIR / "evasion_detection.csv", index=False, float_format="%.4f")
    log("E2 evasion:\n" + evasion.round(4).to_string(index=False))
    ev_tab = evasion.pivot(index="attack_type", columns="model", values="session_detection_rate")
    ev_tab = ev_tab.loc[C.CLASSIC_ATTACKS + C.EVASIVE_ATTACKS, list(fitted)]

    # ------------------------------------------------------------------ E3
    abl_rows, abl_tab = [], {}
    for label, kw in [("30-min sessions", dict(gap_min=C.SESSION_GAP_MIN, max_min=C.SESSION_MAX_MIN)),
                      ("24-h per-IP windows", dict(gap_min=24 * 60, max_min=24 * 60))]:
        sess = sessions if label.startswith("30") else build_session_table(events, **kw)
        Xa_tr, Xa_te, ya_tr, ya_te, _, ia_te = split(sess.reset_index(drop=True), C.SEED)
        m = clone(make_models(C.SEED)["Random Forest"]).fit(Xa_tr, ya_tr)
        s = scores(m, Xa_te)
        abl_rows.append({"window": label, "sessions": len(sess), **binary_metrics(ya_te, s)})
        te_rows = sess.reset_index(drop=True).iloc[ia_te].reset_index(drop=True)
        abl_tab[label] = pd.Series(s >= C.THRESHOLD).groupby(te_rows["kind"]).mean()
    ablation = pd.DataFrame(abl_rows)
    ablation.to_csv(C.RESULTS_DIR / "window_ablation_metrics.csv", index=False, float_format="%.4f")
    kinds = ["human", "service", "stale_device", "stale_credential", *C.CLASSIC_ATTACKS, *C.EVASIVE_ATTACKS]
    abl_tab = pd.DataFrame(abl_tab).reindex([k for k in kinds if k in pd.DataFrame(abl_tab).index])
    abl_tab.to_csv(C.RESULTS_DIR / "window_ablation_by_type.csv", float_format="%.4f")
    log("E3 window ablation:\n" + ablation.round(4).to_string(index=False) + "\n" + abl_tab.round(4).to_string())

    # ------------------------------------------------------------------ E4
    rob_rows = []
    for seed in C.ROBUSTNESS_SEEDS:
        sess = build_session_table(generate_events(seed))
        sess = sess[sess["attack_type"].isin(["benign", *C.CLASSIC_ATTACKS])].reset_index(drop=True)
        a, b, c, d, _, _ = split(sess, seed)
        for name, model in make_models(seed).items():
            m = clone(model).fit(a, c)
            rob_rows.append({"seed": seed, "model": name, **binary_metrics(d, scores(m, b))})
    rob = pd.DataFrame(rob_rows)
    rob.to_csv(C.RESULTS_DIR / "robustness_seeds.csv", index=False, float_format="%.4f")
    rob_sum = rob.groupby("model", sort=False)[cols].agg(["mean", "std"])
    rob_sum.columns = [f"{a}_{b}" for a, b in rob_sum.columns]
    rob_sum.to_csv(C.RESULTS_DIR / "robustness_summary.csv", float_format="%.4f")
    log("E4 robustness (mean over seeds):\n" + rob.groupby("model", sort=False)[cols].mean().round(4).to_string())

    # ------------------------------------------------------------------ figures
    plots.roc_and_pr(yte, test_scores, C.FIG_DIR / "roc_curves.png", C.FIG_DIR / "pr_curves.png")
    plots.confusion_matrices(cms, C.FIG_DIR / "confusion_matrices.png")
    plots.feature_importance(perm, pd.Series(rf.feature_importances_, index=C.FEATURES), coef, C.FIG_DIR / "feature_importance.png")
    plots.failure_ratio_by_class(classic, C.FIG_DIR / "failure_ratio_by_class.png")
    plots.evasion_bars(ev_tab, C.FIG_DIR / "evasion_detection.png",
                       "Detection rate by attack type (models trained on classic attacks only)")
    plots.window_ablation(abl_tab, C.FIG_DIR / "window_ablation.png")

    runtime = time.time() - T0
    json.dump({"seed": C.SEED, "n_events": int(len(events)), "n_sessions": int(len(sessions)),
               "n_classic_sessions": int(len(classic)), "n_test_sessions": int(len(yte)),
               "test_attack_sessions": int(yte.sum()), "runtime_seconds": round(runtime, 1)},
              open(C.RESULTS_DIR / "run_info.json", "w"), indent=2)
    log(f"done - total runtime {runtime:.1f}s")


if __name__ == "__main__":
    main()
