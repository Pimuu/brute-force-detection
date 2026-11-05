# Detecting Brute-Force and Credential-Stuffing Logins from Authentication-Event Features

A small, fully reproducible research project, from code to paper. It classifies per-IP login **sessions** as *attack* or *legitimate*
using seven simple behavioural features, compares a fail2ban-style rule with three scikit-learn
models, and then measures how badly the approach breaks against **evasive** (low-and-slow and
distributed) attacks.

* **Paper:** [`paper/paper.md`](paper/paper.md)
* **Pipeline:** `python run_pipeline.py` (about 55 s on a 4-vCPU container; no downloads, no GPU, no deep learning)
* **Results:** [`results/`](results/) (CSV metrics) and [`results/figures/`](results/figures/) (PNG)

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python run_pipeline.py
```

It is deterministic: `SEED = 42` in `src/config.py` controls everything. Running it again
rewrites `data/` and `results/` with identical numbers. Verified with Python 3.13, numpy 2.5,
pandas 3.0, scikit-learn 1.9, matplotlib 3.11.

## Headline results (seed 42, held-out 30 % test split, 3,214 sessions, 156 attacks)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---|---|---|---|---|---|
| Rule (≥5 failures) | 0.9844 | 0.7650 | 0.9808 | 0.8596 | 0.9979 | 1.54 % |
| Logistic Regression | 0.9956 | 0.9277 | 0.9872 | 0.9565 | 0.9998 | 0.39 % |
| Random Forest | 0.9997 | 1.0000 | 0.9936 | 0.9968 | 0.9997 | 0.00 % |
| HistGradientBoosting | 0.9997 | 1.0000 | 0.9936 | 0.9968 | 0.9993 | 0.00 % |

When the same models face attacks they were not trained on, **low-and-slow guessing is detected in 0 % of
sessions** and **distributed (botnet) stuffing in 0.4 to 21 % of sessions**. Retraining on those attacks restores recall, but
the false-positive rate on benign "stale-device" retries goes up to 17 %. Switching to 24-hour per-IP windows halves that
rate (see paper §4.3).

## Repository layout

```
run_pipeline.py          one-command reproducible pipeline (E1-E4 + figures)
src/config.py            every constant (seed, world size, session gap, features, thresholds)
src/data_generation.py   synthetic authentication-log simulator (documented generative story)
src/features.py          per-IP sessionisation + feature extraction + IP reputation
src/models.py            rule baseline, Logistic Regression, Random Forest, HistGradientBoosting
src/evaluate.py          metric helpers
src/plots.py             all figures
data/                    generated login_events.csv.gz (82,957 events) and sessions.csv.gz
results/                 metrics.csv, cv_metrics.csv, evasion_detection.csv, ... + figures/
paper/paper.md           the paper
```

## Design

### 1. Synthetic data (why and how)

No public dataset combines realistic benign login behaviour, ground-truth attack labels, *and* the
benign look-alikes that cause false positives in practice. So we simulate an organisation and
document every assumption (`src/data_generation.py`, constants in `src/config.py`):

* **History window (14 days, benign only).** Used only to build an IP-reputation prior: how many
  successful logins each IP had.
* **Evaluation window (14 days, benign + attacks).** All sessions used for training and testing come from here.
* **Benign actors.**
  * 600 employees with diurnal schedules (85 % office hours, 10 % night shift, 5 % +8 h time zone).
  * Shared office NAT IPs (many usernames per IP).
  * Dynamic home IPs (2 %/day re-assignment).
  * A 2,000-address carrier-NAT mobile pool (20 % of logins).
  * VPN exits, travel IPs.
  * Typos (12 %) and give-ups (15 % of those).
  * Four service accounts that log in every 10 to 15 min, 24/7.
  * **Hard negatives:**
    * *stale-device retries*: a phone keeps trying an old password after a password change, giving failure-only bursts every 10 to 90 min.
    * *stale-credential incidents*: a service job retries a rotated password every 30 to 60 s for 1 to 6 h.
* **Attacks.**
  * `brute_force` (200 campaigns): 8 to 800 rapid guesses against 1 to 5 usernames.
  * `credential_stuffing` (200): 6 to 600 leaked pairs, 75 % non-existent usernames, about 5 % success on real accounts, sometimes through rotating proxies.
  * `low_and_slow` (25, evasive): one guess every 35 to 180 min.
  * `distributed` (6, evasive): a botnet of 80 to 250 IPs, each making only a few attempts.
  * 15 % of classic campaigns egress through a VPN exit that benign users also use, so the IP is *not* rare.

### 2. Features (per IP, per session)

A session is a run of attempts from one IP with gaps of no more than 30 min, cut at 60 min.

| Feature | Meaning |
|---|---|
| `n_attempts` | attempts in the session |
| `failure_ratio` | failed / total |
| `n_unique_usernames` | distinct usernames tried |
| `attempts_per_minute` | attempts / max(duration, 1 min) |
| `hour_of_day` | hour of first attempt |
| `is_off_hours` | first attempt between 00:00 and 05:59 |
| `is_rare_ip` | fewer than 2 successful logins from this IP in the history window |

Ground-truth provenance (`kind`) is kept only for error analysis. It is never used as a feature.

### 3. Experiments

* **E1:** a stratified 70/30 split plus 5-fold CV on benign + classic attacks.
* **E2:** the E1 models are applied, unchanged, to held-out evasive attacks. We report detection rate per session and per campaign.
* **E3:** 30-min sessions vs. 24-h per-IP windows, with a Random Forest trained on *all* attack types.
* **E4:** E1 repeated on 5 independently generated worlds.

## Threat model

**Defender.** Sees only the authentication log of one service: timestamp, source IP, username, and
success/failure. There is no payload, device fingerprint, CAPTCHA or MFA signal, and no cross-organisation threat
intelligence. Detection runs per source IP, at session granularity, and is evaluated at a fixed
0.5 threshold.

**Adversary goals.** Take over accounts by (a) guessing passwords for a few known or default
usernames (brute force / password guessing, MITRE ATT&CK T1110.001) or (b) replaying leaked
credential pairs across many usernames (credential stuffing, T1110.004).

**Adversary capabilities.**
* *Classic attacker:* runs a tool at machine speed from one or a few IPs, mostly from hosting or
  residential-proxy space that the organisation has never seen, sometimes through a popular VPN.
* *Adaptive attacker (out of distribution for E1):* knows that defenders aggregate per IP and per
  short time window. They either **slow down** below the session gap (low-and-slow) or **spread out**
  over a botnet so that every IP looks like a single mistyped login (distributed).

**Out of scope.** Phishing, MFA fatigue, session hijacking, password spraying across *tenants*,
attackers who already control a known-good internal IP, and poisoning of the history window. We
assume the history window is clean.

## Limitations (short version)

The data is synthetic, so absolute numbers reflect the simulator's assumptions and not any real network.
The generator does not contain every benign quirk that real networks have.
Random session-level splits can place sessions from the same campaign in both train and test. Features carry no identifiers, but this is still optimistic.
See paper §6 for the full discussion.

## License

MIT. The synthetic data contains no real users, IPs or credentials.
