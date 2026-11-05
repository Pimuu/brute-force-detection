"""Per-IP sessionisation and behavioural feature extraction.

A *session* is a maximal run of attempts from one source IP in which consecutive
attempts are at most `gap_min` minutes apart, additionally cut every `max_min`
minutes so that a continuously active IP (e.g. a service account) still yields
bounded, comparable sessions.

Features (one row per session)
------------------------------
n_attempts           number of login attempts in the session
failure_ratio        failed / total attempts
n_unique_usernames   distinct usernames tried from the IP in the session
attempts_per_minute  n_attempts / max(duration in minutes, 1)
hour_of_day          hour (0-23) of the first attempt
is_off_hours         1 if hour_of_day in OFF_HOURS (default 00:00-05:59)
is_rare_ip           1 if the IP had < RARE_IP_MIN_SUCCESSES successful logins in the
                     benign *history* window (a simple IP-reputation prior)

Labels: is_attack = 1 if any attempt in the session belongs to an attack
campaign; attack_type is the campaign type (or "benign"); kind is the
ground-truth actor (benign sub-type or attack type) used only for error analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


ACTOR_PRIORITY = {"human": 1, "service": 2, "stale_device": 3, "stale_credential": 4,
                  "low_and_slow": 5, "distributed": 6, "credential_stuffing": 7, "brute_force": 8}


def ip_reputation(history: pd.DataFrame) -> pd.Series:
    """Successful-login count per IP during the (benign) history window."""
    return history.loc[history["success"]].groupby("ip").size()


def sessionize(events: pd.DataFrame, gap_min: float = C.SESSION_GAP_MIN,
               max_min: float = C.SESSION_MAX_MIN) -> pd.Series:
    """Return an integer session id for every event (vectorised)."""
    ev = events.sort_values(["ip", "timestamp"], kind="mergesort")
    new_ip = ev["ip"].ne(ev["ip"].shift())
    gap = ev["timestamp"].diff().dt.total_seconds().div(60)
    gap_sid = (new_ip | (gap > gap_min)).cumsum()
    # split sessions longer than max_min into fixed-length chunks
    start = ev.groupby(gap_sid)["timestamp"].transform("min")
    chunk = ((ev["timestamp"] - start).dt.total_seconds() // (max_min * 60)).astype(int)
    sid = pd.factorize(pd.MultiIndex.from_arrays([gap_sid, chunk]))[0]
    return pd.Series(sid, index=ev.index).reindex(events.index)


def extract_features(events: pd.DataFrame, reputation: pd.Series,
                     gap_min: float = C.SESSION_GAP_MIN,
                     max_min: float = C.SESSION_MAX_MIN) -> pd.DataFrame:
    """Aggregate an event log into one feature row per (IP, session)."""
    ev = events.copy()
    ev["session_id"] = sessionize(ev, gap_min, max_min)
    ev["fail"] = (~ev["success"]).astype(int)
    # attack_type of a session = type of its attack events (benign if none)
    ev["atype_rank"] = np.where(ev["is_attack"] == 1, ev["attack_type"], "")
    # provenance for error analysis: the most "suspicious-looking" actor in the session
    ev["actor_rank"] = ev["actor"].map(ACTOR_PRIORITY).fillna(0)

    g = ev.groupby("session_id")
    s = pd.DataFrame({
        "ip": g["ip"].first(),
        "start": g["timestamp"].min(),
        "end": g["timestamp"].max(),
        "n_attempts": g.size(),
        "n_failures": g["fail"].sum(),
        "n_unique_usernames": g["username"].nunique(),
        "is_attack": g["is_attack"].max(),
        "attack_type": g["atype_rank"].max(),
        "campaign_id": g["campaign_id"].max(),
        "actor_rank": g["actor_rank"].max(),
    })
    s["attack_type"] = s["attack_type"].replace("", "benign")
    s["kind"] = s.pop("actor_rank").map({v: k for k, v in ACTOR_PRIORITY.items()})
    s["failure_ratio"] = s["n_failures"] / s["n_attempts"]
    duration_min = (s["end"] - s["start"]).dt.total_seconds() / 60
    s["duration_min"] = duration_min
    s["attempts_per_minute"] = s["n_attempts"] / np.maximum(duration_min, 1.0)
    s["hour_of_day"] = s["start"].dt.hour
    lo, hi = C.OFF_HOURS
    s["is_off_hours"] = s["hour_of_day"].between(lo, hi - 1).astype(int)
    known = reputation.reindex(s["ip"]).fillna(0).to_numpy()
    s["is_rare_ip"] = (known < C.RARE_IP_MIN_SUCCESSES).astype(int)
    return s.reset_index()


def build_session_table(events: pd.DataFrame, **kw) -> pd.DataFrame:
    """History window -> IP reputation; evaluation window -> labelled sessions."""
    hist = events[events["window"] == "history"]
    evalw = events[events["window"] == "eval"]
    return extract_features(evalw, ip_reputation(hist), **kw)
