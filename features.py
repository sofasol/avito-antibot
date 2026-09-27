"""Признаки по кукам: набор функций, каждая строит одну группу признаков из
событий (events.csv.gz) — объём, тайминг, поиск, курсор, популярность объявлений
и так далее.
"""

import re

import numpy as np
import pandas as pd

PLATFORM_MAP = {
    "web": "web", "WEB": "web", "Web": "web", "desktop": "web",
    "android": "android", "Android": "android", "ANDROID": "android",
    "ios": "ios", "iOS": "ios", "IOS": "ios", "iphone": "ios",
}

SCRIPT_UA = r"Headless|Scrapy|curl|Go-http-client|python|node-fetch"


def prepare_events(events, meta):
    ev = events.drop_duplicates().copy()
    ev["platform"] = ev["platform"].map(PLATFORM_MAP)
    ev = ev.merge(meta[["cookie_id", "window_start_ts", "window_end_ts"]], on="cookie_id")
    ev = ev[(ev["event_ts"] >= ev["window_start_ts"]) & (ev["event_ts"] < ev["window_end_ts"])]
    ev = ev.sort_values(["cookie_id", "event_ts"]).reset_index(drop=True)
    assert ev["platform"].notna().all() and "captcha_shown" not in set(ev["event_name"])
    return ev


def feats_volume(ev):
    n = ev.groupby("cookie_id").size().rename("n_events")
    share = pd.crosstab(ev["cookie_id"], ev["event_name"], normalize="index").add_prefix("share_")
    return pd.concat([n, share], axis=1)


def feats_timing(ev):
    cookie = ev["cookie_id"]
    gap = ev.groupby("cookie_id")["event_ts"].diff().dt.total_seconds().groupby(cookie)
    out = pd.DataFrame({"median_gap_s": gap.median(), "gap_cv": gap.std() / gap.mean()})
    out["gap_cv"] = out["gap_cv"].replace(np.inf, np.nan)
    per_min = ev.groupby(["cookie_id", ev["event_ts"].dt.floor("min")]).size()
    out["peak_per_min"] = per_min.groupby("cookie_id").max()
    out["night_share"] = (ev["event_ts"].dt.hour < 6).groupby(cookie).mean()
    return out


def feats_search(ev):
    g = ev[ev["event_name"] == "search_results_view"].groupby("cookie_id")
    out = pd.DataFrame({
        "n_search": g.size(), "n_queries": g["search_query"].nunique(), "max_page": g["search_page"].max(),
    })
    out["queries_per_search"] = out["n_queries"] / out["n_search"]
    return out


def feats_diversity(ev):
    views = ev[ev["event_name"] == "item_view"]
    g = views.groupby("cookie_id")
    out = pd.DataFrame({
        "n_items": g["item_id"].nunique(), "n_categories": g["item_category"].nunique(),
        "n_locations": g["item_location"].nunique(),
    })
    out["repeat_views"] = 1 - out["n_items"] / g.size()
    per_cat = views.groupby(["cookie_id", "item_category"]).size()
    out["top_cat_share"] = per_cat.groupby("cookie_id").max() / per_cat.groupby("cookie_id").sum()
    return out


def feats_pointer(ev):
    web = ev[ev["platform"] == "web"]
    out = pd.DataFrame({"ptr_share": web["pointer_x"].notna().groupby(web["cookie_id"]).mean()})
    xy = web.groupby("cookie_id")[["pointer_x", "pointer_y"]].agg(["mean", "std"])
    xy.columns = ["ptr_x_mean", "ptr_x_std", "ptr_y_mean", "ptr_y_std"]
    return out.join(xy)


def feats_tech(ev):
    ua_kind = np.select(
        [ev["user_agent"].str.contains(SCRIPT_UA, case=False), ev["user_agent"].str.startswith("Avito/")],
        ["script", "app"], default="browser",
    )
    ev = ev.assign(ua_kind=ua_kind)
    top = lambda s: s.mode().iat[0]
    g = ev.groupby("cookie_id")
    chrome = ev["user_agent"].str.extract(r"Chrome/(\d+)")[0].astype(float).groupby(ev["cookie_id"]).median()
    return pd.DataFrame({"platform": g["platform"].agg(top), "ua_kind": g["ua_kind"].agg(top), "chrome_version": chrome})


def feats_age(meta):
    return ((meta["window_start_ts"] - meta["cookie_created_at"]).dt.days).set_axis(meta["cookie_id"]).rename("age_days").to_frame()


def feats_popularity(ev, K=3):
    views = ev[ev["event_name"] == "item_view"][["cookie_id", "item_id", "window_start_ts"]].copy()
    d = (views["window_start_ts"] - pd.Timestamp("2026-04-06")).dt.days.values
    per_day = views.assign(d=d).groupby(["item_id", "d"])["cookie_id"].nunique().unstack(fill_value=0).reindex(columns=range(21), fill_value=0)
    cum = per_day.cumsum(axis=1).values
    rows = per_day.index.get_indexer(views["item_id"])
    lo = np.where(d - K - 1 >= 0, cum[rows, np.maximum(d - K - 1, 0)], 0)
    other = pd.Series(cum[rows, d] - lo - 1)
    cookie = views["cookie_id"].values
    return pd.DataFrame({"pop_mean": np.log1p(other).groupby(cookie).mean(), "pop_max": other.groupby(cookie).max(),
                         "pop_shared": (other > 0).astype(float).groupby(cookie).mean()})


def feats_timing2(ev):
    c = ev["cookie_id"]
    gap = ev.groupby("cookie_id")["event_ts"].diff().dt.total_seconds()
    g = gap.groupby(c)
    out = pd.DataFrame({
        "gap_p10": g.quantile(.1), "gap_p25": g.quantile(.25), "gap_p75": g.quantile(.75), "gap_p90": g.quantile(.9),
        "gap_min": g.min(), "gap_max": g.max(), "gap_mean": g.mean(), "gap_std": g.std(),
    })
    for t in (5, 15, 30, 60, 120):
        out[f"gap_lt{t}"] = (gap < t).astype(float).where(gap.notna()).groupby(c).mean()
    bounds = ev.groupby("cookie_id")["event_ts"].agg(["min", "max"])
    span = (bounds["max"] - bounds["min"]).dt.total_seconds() / 60
    out["events_per_min"] = ev.groupby("cookie_id").size() / (span + 1)
    out["n_hours"] = ev["event_ts"].dt.hour.groupby(c).nunique()
    return out


def feats_cursor2(ev):
    w = ev[(ev["platform"] == "web") & ev["pointer_x"].notna()]
    g = w.groupby("cookie_id")
    cookie = w["cookie_id"]
    out = pd.DataFrame({f"px_q{int(q * 100)}": g["pointer_x"].quantile(q) for q in (.1, .25, .5, .75, .9)})
    for q in (.1, .5, .9):
        out[f"py_q{int(q * 100)}"] = g["pointer_y"].quantile(q)
    out["px_min"], out["px_max"], out["px_skew"] = g["pointer_x"].min(), g["pointer_x"].max(), g["pointer_x"].skew()
    out["left_share"] = (w["pointer_x"] < 300).groupby(cookie).mean()
    out["right_share"] = (w["pointer_x"] > 1620).groupby(cookie).mean()
    out["edge_share"] = ((w["pointer_x"] <= 5) | (w["pointer_x"] >= 1915) | (w["pointer_y"] <= 5) | (w["pointer_y"] >= 1075)).groupby(cookie).mean()
    out["n_ptr"] = g.size()
    out["step_mean"] = np.hypot(g["pointer_x"].diff(), g["pointer_y"].diff()).groupby(cookie).mean()
    out["xy_corr"] = g[["pointer_x", "pointer_y"]].apply(lambda d: d["pointer_x"].corr(d["pointer_y"]) if len(d) >= 3 else np.nan)
    return out


def parse_ua(u):
    m = re.match(r"Avito/(\d+)\.0 \((?:Android (\d+); ([^)]+)|iPhone; iOS ([\d.]+);)", u)
    if m:
        return ("app", float(m.group(1)), float(m.group(2) or m.group(4)), m.group(3) or "iPhone")
    m = re.search(r"Android (\d+); ([^)]+)\).*Chrome/(\d+)", u)
    if m:
        return ("chrome_m", np.nan, float(m.group(1)), m.group(2))
    if "iPhone" in u:
        return ("safari_ios", np.nan, np.nan, "iPhone")
    return ("web", np.nan, np.nan, "desktop")


def feats_ua2(ev):
    uas = ev["user_agent"].drop_duplicates()
    parts = pd.DataFrame([parse_ua(u) for u in uas], index=uas, columns=["ua_form", "app_version", "os_version", "device"])
    top = lambda s: s.dropna().mode().iat[0] if s.notna().any() else np.nan
    return ev[["cookie_id", "user_agent"]].join(parts, on="user_agent").groupby("cookie_id")[list(parts.columns)].agg(top)
