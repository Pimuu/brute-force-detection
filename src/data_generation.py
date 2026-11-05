"""Synthetic authentication-log generator.

The generator simulates a mid-sized organisation's login endpoint for
HISTORY_DAYS + EVAL_DAYS days and emits one row per *login attempt*:

    timestamp, ip, username, success, is_attack, attack_type, campaign_id, window, actor

`actor` is ground-truth provenance (human, stale_device, service,
stale_credential, or the attack type). It is used only for error analysis and
is never given to a model.

Generative story
----------------
Benign traffic (both windows)
  * N_USERS employees. Each has a home IP; ~65 % also work from one of a few
    corporate NAT egress IPs that many colleagues share (a realistic source of
    "many usernames per IP" that is *not* an attack).
  * Daily activity: ~85 % of users keep office hours, 10 % work nights, 5 % live
    in a time zone 8 h ahead. Weekday activity is higher than weekend.
  * Home IPs are dynamic: each day a user's residential IP is re-assigned with
    probability P_HOME_IP_CHURN, so a once-known user can appear from a "rare" IP.
  * ~20 % of logins come from phones behind a large carrier-NAT pool, a small
    share from never-seen IPs (travel, hotel Wi-Fi) or shared commercial-VPN exits.
  * 12 % of logins start with 1-8 typos; 8 % of those users give up.
  * Stale-device retries: after a password change, a user's forgotten phone /
    mail client keeps re-trying the *old* password in small bursts every
    10-90 min for hours - failure-only sessions from mostly mobile IPs that look
    very much like a low-and-slow guessing attack (a classic SOC false positive).
  * Service accounts authenticate every 10-15 min, 24/7, from internal IPs.
  * "Stale credential" incidents: after a password rotation a job retries the old
    password every 30-60 s for hours -> a dense failure burst that is benign.

Attack traffic (evaluation window only)
  * brute_force: one IP hammers 1-5 usernames (real or default names such as
    "admin") with many rapid guesses; very rarely the last guess succeeds.
  * credential_stuffing: one IP (or a few rotating proxies) replays a leaked
    username/password list - many distinct usernames, mostly non-existent,
    each tried ~once, ~5 % success on real accounts due to password reuse.
  * low_and_slow (evasive): one guess every 35-180 minutes for days.
  * distributed (evasive): a botnet of 80-250 IPs shares a credential list so
    every IP only makes a handful of attempts.
  * 15 % of classic campaigns egress through a VPN exit that benign users also
    use, so the IP is *not* rare.

Everything is driven by a single numpy Generator so the dataset is exactly
reproducible from SEED.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

COMMON_TARGETS = ["admin", "root", "test", "administrator", "guest", "info",
                  "support", "oracle", "user", "backup", "webmaster", "sysadmin"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class IPAllocator:
    """Hands out unique dotted-quad IPs from disjoint address spaces."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.used: set[str] = set()

    def new(self, first_octets: tuple[int, ...]) -> str:
        while True:
            a = int(self.rng.choice(first_octets))
            ip = f"{a}.{self.rng.integers(0, 256)}.{self.rng.integers(0, 256)}.{self.rng.integers(1, 255)}"
            if ip not in self.used:
                self.used.add(ip)
                return ip


RESIDENTIAL = (24, 50, 67, 73, 98)          # benign home / travel space
MOBILE = (166, 174)                           # carrier-grade NAT pool for phones
OFFICE = (198,)
INTERNAL = (10,)
VPN = (185,)
ATTACKER = (5, 45, 91, 103, 141, 193, 212)    # hosting / bulletproof / botnet space


def _attempt_sequence(rng, t0: float, n_fail: int, succeed: bool) -> list[tuple[float, bool]]:
    """Benign human login: n_fail typos (4-40 s apart) then optionally a success."""
    out, t = [], t0
    for _ in range(n_fail):
        out.append((t, False))
        t += 4 + rng.exponential(12)
    if succeed:
        out.append((t, True))
    return out


# --------------------------------------------------------------------------- #
# Benign traffic
# --------------------------------------------------------------------------- #
def _make_users(rng, ipa: IPAllocator):
    offices = [ipa.new(OFFICE) for _ in range(C.N_OFFICES)]
    users = []
    for i in range(C.N_USERS):
        sched = rng.choice(["day", "night", "shifted"], p=[0.85, 0.10, 0.05])
        users.append(dict(
            username=f"emp{i:04d}",
            home_ip=ipa.new(RESIDENTIAL),
            office_ip=offices[rng.integers(C.N_OFFICES)] if rng.random() < C.P_OFFICE_USER else None,
            first_hour={"day": 8.5, "night": 21.5, "shifted": 16.5}[sched],
        ))
    return users, offices


def _benign_events(rng, ipa, users, vpn_exits, mobile_pool, n_days: int, day_offset: int, window: str):
    rows = []
    for d in range(n_days):
        day = day_offset + d
        weekday = (day % 7) < 5
        for u in users:
            if rng.random() < C.P_HOME_IP_CHURN:
                u["home_ip"] = ipa.new(RESIDENTIAL)                 # DHCP re-assignment
            if rng.random() > (0.85 if weekday else 0.25):
                continue
            n_logins = 1 + rng.poisson(0.8)
            first = rng.normal(u["first_hour"], 1.5)
            hours = [first] + list(first + rng.uniform(0.5, 9.0, n_logins - 1))
            for h in hours:
                h = h % 24
                t0 = day * 86400 + h * 3600
                # choose source IP
                r = rng.random()
                if r < C.P_BENIGN_RARE_IP:
                    ip = ipa.new(RESIDENTIAL)                       # travel / mobile
                elif r < C.P_BENIGN_RARE_IP + C.P_BENIGN_VIA_VPN:
                    ip = vpn_exits[rng.integers(len(vpn_exits))]
                elif r < C.P_BENIGN_RARE_IP + C.P_BENIGN_VIA_VPN + C.P_BENIGN_MOBILE:
                    ip = mobile_pool[rng.integers(len(mobile_pool))]
                elif u["office_ip"] and weekday and 7 <= h <= 19 and rng.random() < 0.8:
                    ip = u["office_ip"]
                else:
                    ip = u["home_ip"]
                # typos / give-ups
                n_fail, succeed = 0, True
                if rng.random() < C.P_TYPO:
                    n_fail = min(int(rng.geometric(0.55)), 8)
                    succeed = rng.random() > C.P_GIVE_UP
                for t, ok in _attempt_sequence(rng, t0, n_fail, succeed):
                    rows.append((t, ip, u["username"], ok, 0, "benign", -1, window))
    return rows


def _stale_device_retries(rng, users, mobile_pool, n_days: int, day_offset: int, window: str):
    """Old password cached on a second device keeps failing after a password change."""
    rows = []
    for u in users:
        if rng.random() > C.P_PASSWORD_CHANGE * C.P_SECOND_DEVICE:
            continue
        t = (day_offset + rng.uniform(0, n_days)) * 86400
        end = t + rng.uniform(1, 36) * 3600
        ip = mobile_pool[rng.integers(len(mobile_pool))] if rng.random() < 0.7 else u["home_ip"]
        while t < end:
            for k in range(int(rng.integers(1, 4))):            # 1-3 quick retries per sync
                rows.append((t + k * rng.uniform(2, 10), ip, u["username"], False, 0,
                             "benign", -1, window))
            t += rng.uniform(10, 90) * 60
            if rng.random() < 0.3 and ip in mobile_pool:
                ip = mobile_pool[rng.integers(len(mobile_pool))]   # phone hops carrier IPs
    return rows


def _service_events(rng, ipa, n_days: int, day_offset: int, window: str, accounts):
    rows = []
    for acc in accounts:
        period = acc["period_s"]
        t = day_offset * 86400 + rng.uniform(0, period)
        end = (day_offset + n_days) * 86400
        while t < end:
            rows.append((t, acc["ip"], acc["username"], True, 0, "benign", -1, window))
            t += period + rng.normal(0, 20)
    return rows


def _stale_credential_incidents(rng, accounts, day_offset: int, n_days: int, window: str):
    """A job keeps retrying an outdated password for 1-6 h (benign, but looks hostile)."""
    rows = []
    for _ in range(C.N_STALE_CRED_INCIDENTS):
        acc = accounts[rng.integers(len(accounts))]
        t = (day_offset + rng.uniform(0, n_days)) * 86400
        end = t + rng.uniform(1, 6) * 3600
        while t < end:
            rows.append((t, acc["ip"], acc["username"], False, 0, "benign", -1, window))
            t += rng.uniform(30, 60)
    return rows


# --------------------------------------------------------------------------- #
# Attack traffic
# --------------------------------------------------------------------------- #
def _attack_start(rng, day_offset: int, n_days: int) -> float:
    day = day_offset + rng.integers(0, n_days)
    hour = rng.uniform(0, 6) if rng.random() < 0.5 else rng.uniform(0, 24)
    return day * 86400 + hour * 3600


def _attacker_ip(rng, ipa, vpn_exits) -> str:
    if rng.random() < C.P_ATTACK_VIA_VPN:
        return vpn_exits[rng.integers(len(vpn_exits))]
    return ipa.new(ATTACKER)


def _fake_usernames(rng, n: int) -> list[str]:
    stems = ["john", "maria", "alex", "sam", "lee", "chris", "anna", "mike", "kate", "tom",
             "nina", "omar", "li", "yuki", "raj", "ivan", "sara", "paul", "eva", "dan"]
    return [f"{stems[rng.integers(len(stems))]}{rng.integers(1, 9999)}" for _ in range(n)]


def _brute_force(rng, ipa, users, vpn_exits, day_offset, n_days, cid):
    ip = _attacker_ip(rng, ipa, vpn_exits)
    n_targets = 1 if rng.random() < 0.6 else int(rng.integers(2, 6))
    targets = [users[rng.integers(len(users))]["username"] if rng.random() < 0.5
               else COMMON_TARGETS[rng.integers(len(COMMON_TARGETS))] for _ in range(n_targets)]
    n = int(np.clip(rng.lognormal(np.log(60), 1.0), 8, 800))
    mean_gap = np.exp(rng.uniform(np.log(0.5), np.log(20)))      # tool speed, seconds
    t = _attack_start(rng, day_offset, n_days) + np.cumsum(rng.exponential(mean_gap, n))
    names = rng.choice(targets, n)
    ok = np.zeros(n, bool)
    if rng.random() < 0.08 and names[-1].startswith("emp"):
        ok[-1] = True                                          # password finally guessed
    return [(t[i], ip, names[i], bool(ok[i]), 1, "brute_force", cid, "eval") for i in range(n)]


def _credential_stuffing(rng, ipa, users, vpn_exits, day_offset, n_days, cid,
                         n_creds=None, ips=None, spread_s=None, kind="credential_stuffing"):
    if n_creds is None:
        n_creds = int(np.clip(rng.lognormal(np.log(50), 1.0), 6, 600))
    if ips is None:
        ips = [_attacker_ip(rng, ipa, vpn_exits) for _ in range(1 + rng.poisson(0.5))]
    n_real = rng.binomial(n_creds, 0.25)
    names = [users[i]["username"] for i in rng.integers(0, len(users), n_real)]
    names += _fake_usernames(rng, n_creds - n_real)
    rng.shuffle(names)
    names += [n for n in names if rng.random() < 0.10]          # some pairs retried
    n = len(names)
    if spread_s is None:
        gaps = rng.exponential(np.exp(rng.uniform(np.log(1), np.log(30))), n)
    else:
        gaps = rng.exponential(spread_s / n, n)
    t = _attack_start(rng, day_offset, n_days) + np.cumsum(gaps)
    src = [ips[i] for i in rng.integers(0, len(ips), n)]
    ok = [nm.startswith("emp") and rng.random() < 0.05 for nm in names]
    return [(t[i], src[i], names[i], ok[i], 1, kind, cid, "eval") for i in range(n)]


def _low_and_slow(rng, ipa, users, day_offset, n_days, cid):
    ip = ipa.new(ATTACKER)
    targets = [users[rng.integers(len(users))]["username"] for _ in range(rng.integers(1, 4))]
    n = int(rng.integers(20, 81))
    t = day_offset * 86400 + rng.uniform(0, n_days * 0.5) * 86400 + np.cumsum(rng.uniform(35, 180, n) * 60)
    end = (day_offset + n_days) * 86400
    return [(t[i], ip, targets[rng.integers(len(targets))], False, 1, "low_and_slow", cid, "eval")
            for i in range(n) if t[i] < end]


def _distributed(rng, ipa, users, day_offset, n_days, cid):
    bots = [ipa.new(ATTACKER) for _ in range(rng.integers(80, 251))]
    return _credential_stuffing(rng, ipa, users, None, day_offset, n_days, cid,
                                n_creds=int(rng.integers(150, 601)), ips=bots,
                                spread_s=rng.uniform(2, 12) * 3600, kind="distributed")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_events(seed: int = C.SEED) -> pd.DataFrame:
    """Return the full event log (history + evaluation windows) as a DataFrame."""
    rng = np.random.default_rng(seed)
    ipa = IPAllocator(rng)
    users, _ = _make_users(rng, ipa)
    vpn_exits = [ipa.new(VPN) for _ in range(C.N_VPN_EXITS)]
    mobile_pool = [ipa.new(MOBILE) for _ in range(C.N_MOBILE_IPS)]
    accounts = [dict(username=f"svc_{name}", ip=ipa.new(INTERNAL), period_s=rng.uniform(600, 900))
                for name in ["backup", "ci", "monitor", "etl"][:C.N_SERVICE_ACCOUNTS]]

    H, E = C.HISTORY_DAYS, C.EVAL_DAYS
    rows, actors = [], []

    def add(chunk, actor):
        """Append generated rows and remember who produced them (ground-truth provenance)."""
        rows.extend(chunk)
        actors.extend([actor] * len(chunk))

    for win, off, n in [("history", 0, H), ("eval", H, E)]:
        add(_benign_events(rng, ipa, users, vpn_exits, mobile_pool, n, off, win), "human")
        add(_stale_device_retries(rng, users, mobile_pool, n, off, win), "stale_device")
        add(_service_events(rng, ipa, n, off, win, accounts), "service")
    add(_stale_credential_incidents(rng, accounts, H, E, "eval"), "stale_credential")

    cid = 0
    for _ in range(C.N_BRUTE_FORCE):
        add(_brute_force(rng, ipa, users, vpn_exits, H, E, cid), "brute_force"); cid += 1
    for _ in range(C.N_CRED_STUFFING):
        add(_credential_stuffing(rng, ipa, users, vpn_exits, H, E, cid), "credential_stuffing"); cid += 1
    for _ in range(C.N_LOW_AND_SLOW):
        add(_low_and_slow(rng, ipa, users, H, E, cid), "low_and_slow"); cid += 1
    for _ in range(C.N_DISTRIBUTED):
        add(_distributed(rng, ipa, users, H, E, cid), "distributed"); cid += 1

    df = pd.DataFrame(rows, columns=["t", "ip", "username", "success", "is_attack",
                                     "attack_type", "campaign_id", "window"])
    df["actor"] = actors
    # attacks that spill past the end of the window are truncated
    df = df[df["t"] < (H + E) * 86400].copy()
    df["timestamp"] = pd.Timestamp(C.START_DATE) + pd.to_timedelta(df["t"], unit="s")
    df = df.drop(columns="t").sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df[["timestamp", "ip", "username", "success", "is_attack", "attack_type",
               "campaign_id", "window", "actor"]]
