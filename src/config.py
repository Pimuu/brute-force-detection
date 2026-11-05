"""Central configuration for the brute-force / credential-stuffing detection pipeline.

Every tunable constant lives here so that experiments are reproducible and the
paper can cite exact values. Changing SEED regenerates a different (but
statistically identical) synthetic world.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
ROBUSTNESS_SEEDS = [0, 1, 2, 3, 4]  # independent re-generations of the world

# --------------------------------------------------------------------------- #
# Simulated world (see src/data_generation.py for the generative story)
# --------------------------------------------------------------------------- #
START_DATE = "2026-01-05"        # a Monday
HISTORY_DAYS = 14                # benign-only baseline used for IP reputation
EVAL_DAYS = 14                   # benign + attack traffic; models are trained/tested here

N_USERS = 600
N_OFFICES = 6                    # corporate NAT egress IPs shared by many users
P_OFFICE_USER = 0.65
N_VPN_EXITS = 5                  # commercial-VPN egress IPs shared by benign users and attackers
P_BENIGN_VIA_VPN = 0.04
P_BENIGN_RARE_IP = 0.04          # travel / hotel Wi-Fi -> never-seen IP
P_BENIGN_MOBILE = 0.20           # phone logins via a carrier-NAT pool
N_MOBILE_IPS = 2000              # size of that pool (large -> many pool IPs are "rare")
P_HOME_IP_CHURN = 0.02           # daily probability a user's home IP is re-assigned
P_PASSWORD_CHANGE = 0.08         # per user per window: password changed ...
P_SECOND_DEVICE = 0.60           # ... and a forgotten device keeps retrying the old one
P_TYPO = 0.12                    # a benign login starts with >=1 failed attempt
P_GIVE_UP = 0.15                 # ... and the user gives up instead of succeeding

N_SERVICE_ACCOUNTS = 4           # automated jobs logging in every 10-15 min, 24/7
N_STALE_CRED_INCIDENTS = 3       # a job keeps retrying a rotated password (benign!)

# Attack campaigns (all inside the evaluation window)
N_BRUTE_FORCE = 200
N_CRED_STUFFING = 200
N_LOW_AND_SLOW = 25              # evasive: held out of training in Experiment 2
N_DISTRIBUTED = 6                # evasive: botnet spreading attempts over many IPs
P_ATTACK_VIA_VPN = 0.15          # attacker hides behind a VPN exit benign users also use

CLASSIC_ATTACKS = ["brute_force", "credential_stuffing"]
EVASIVE_ATTACKS = ["low_and_slow", "distributed"]

# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
SESSION_GAP_MIN = 30             # inactivity gap that closes a per-IP session
SESSION_MAX_MIN = 60             # hard cap on session length
RARE_IP_MIN_SUCCESSES = 2        # IP is "rare" if < this many successful logins in history
OFF_HOURS = (0, 6)               # [start, end) local hours considered unusual

FEATURES = [
    "n_attempts",
    "failure_ratio",
    "n_unique_usernames",
    "attempts_per_minute",
    "hour_of_day",
    "is_off_hours",
    "is_rare_ip",
]
COUNT_FEATURES = ["n_attempts", "n_unique_usernames", "attempts_per_minute"]  # log1p for LR

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
TEST_SIZE = 0.30
CV_FOLDS = 5
THRESHOLD = 0.5
RULE_MIN_FAILURES = 5            # fail2ban-style baseline: >= 5 failures in a session
