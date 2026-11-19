"""Forecast GTA VI trophy earn rates (Exophase) at a case target.

Deterministic, offline, stdlib-only. For each trophy it forecasts five
quantiles (0.05 / 0.25 / 0.50 / 0.75 / 0.95) of its Exophase earn rate --
the percent of tracked players who have earned it -- at the case target.

Why the model looks the way it does
------------------------------------
An earn rate is a ratio, r = E / N (earners over players Exophase tracks).
N grows as the launch reaches more players; E grows only as players reach
that trophy's content. Exophase refreshes a game's counts on its own
schedule, hours apart -- between refreshes the page is frozen, and the
answer is the page as of its last refresh before the target.

Two regimes, decided by whether the history ever moved:

* Evolving history (distinct reads exist): per trophy, a least-squares
  line of logit(rate) against ln(1 + hours since launch), extrapolated to
  the target with the move from the last read damped to 70% (trends
  decelerate: the cohort ages, late joiners are slower, story completion
  saturates). The spread grows with the residual scatter, the horizon,
  and a floor for model error.
* Frozen history (every read identical): the page is stale and a refresh
  is overdue. A refresh dilutes every rate except the one at 100% -- new
  players are tracked before earning anything. At the first refresh on
  launch night N rose 6.06x while non-100% earners rose only 1.34x, so
  the average non-100% rate fell to ~0.22x. The median is the last rate
  times a factor that keeps more of the rate for trophies the earliest
  players earned fastest, with a wide spread because whether the refresh
  lands before the target, and how far N jumps, is genuinely uncertain.

Reads every ``*.in`` file in the input dir (name order) and writes one
``.out`` per file. Input/output dirs come from MAKEFUN_INPUT_DIR /
MAKEFUN_OUTPUT_DIR, else /input / /output. No randomness, no network,
no timestamps in the output: two runs are byte-identical.
"""

import json
import math
import os
from datetime import datetime, timezone

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)

RATE_MIN = 0.01
RATE_MAX = 99.99

WIDE_PRIOR = [0.01, 0.05, 0.5, 5.0, 50.0]


def clamp_rate(x):
    """Clamp a rate in percent to the scorer's [0.01, 99.99] range."""
    return min(RATE_MAX, max(RATE_MIN, x))


def logit(rate):
    """Logit of a rate given in percent."""
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(z):
    """Inverse logit, returned as a rate in percent."""
    if z >= 0:
        return 100.0 / (1.0 + math.exp(-z))
    return 100.0 * math.exp(z) / (1.0 + math.exp(z))


def parse_time(text):
    """Parse an ISO-8601 timestamp (trailing Z allowed)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_launch(when):
    """Hours between launch (2026-11-19T00:00Z) and a datetime."""
    return (when - LAUNCH).total_seconds() / 3600.0


def distinct_reads(history):
    """Collapse consecutive identical reads to one point each.

    Exophase freezes the page between refreshes, so long runs of reads
    carry the same rates; counting each separately would overweight a
    stale plateau in the trend fit. Null rates are kept in the tuple so
    a read that only differs by a null still counts as a new point.
    """
    out = []
    prev = None
    for read in history or []:
        key = tuple(read.get("rates") or [])
        if key != prev:
            out.append(read)
            prev = key
    return out


def fit_line(xs, ys):
    """Least-squares line y = a + b*x; returns (a, b, resid).

    resid is the root-mean-square of the residuals (0 when fewer than
    two points, where no line can be fit).
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return ys[0], 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, resid


def quantiles_from_logit(median_logit, sigma):
    """Five rate quantiles around a logit-space median, in percent."""
    qs = []
    for z in Z:
        q = inv_logit(median_logit + z * sigma)
        qs.append(round(clamp_rate(q), 4))
    # Rounding can only tie neighbours, never invert them, but enforce
    # the non-decreasing order the schema requires all the same.
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_trophy_stale(last_rate):
    """Frozen history: the page is stale and a refresh is overdue.

    A refresh dilutes every rate except 100%: new players are tracked
    before earning anything. At the first launch-night refresh the
    average non-100% rate fell to ~0.22x, so the median is the last
    rate times a factor around that, keeping more for trophies the
    earliest players earned fastest (their earners keep growing too).
    """
    if last_rate is None:
        return list(WIDE_PRIOR)
    if last_rate >= 100.0:
        return [RATE_MAX] * 5
    if last_rate <= 0.0:
        return [0.01, 0.01, 0.01, 0.05, 0.5]
    factor = 0.25 * math.sqrt(last_rate / 0.78)
    factor = min(0.60, max(0.15, factor))
    median_logit = logit(last_rate * factor)
    return quantiles_from_logit(median_logit, 0.9)


def forecast_trophy_evolving(points, horizon_days, target_x):
    """Evolving history: logit-linear trend, damped extrapolation.

    points is a list of (x, rate) with x = ln(1 + hours since launch),
    in read order. The extrapolated move from the last read's logit is
    damped to 70%; the spread grows with residual scatter, horizon,
    and a model-error floor, capped at 1.6 logits. With fewer than
    three points there is no trend to trust, so the last rate is held
    with a horizon-wide spread.
    """
    last_rate = points[-1][1]
    if all(rate >= 100.0 for _, rate in points):
        return [RATE_MAX] * 5
    if len(points) < 3:
        sigma = 0.5 * math.sqrt(max(horizon_days, 0.0)) + 0.25
        return quantiles_from_logit(logit(last_rate), sigma)
    xs = [p[0] for p in points]
    ys = [logit(p[1]) for p in points]
    a, b, resid = fit_line(xs, ys)
    b = min(3.0, max(-3.0, b))
    last_logit = logit(last_rate)
    pred = last_logit + 0.7 * ((a + b * target_x) - last_logit)
    sigma = math.sqrt(resid ** 2
                      + (0.40 * math.sqrt(max(horizon_days, 0.0))) ** 2
                      + 0.15 ** 2)
    sigma = min(sigma, 1.6)
    return quantiles_from_logit(pred, sigma)


def forecast_case(case):
    """Forecast every trophy of one case; returns the output dict."""
    history = distinct_reads(case.get("history") or [])
    horizon_days = case.get("horizon_minutes", 0) / 1440.0
    target_x = math.log1p(hours_since_launch(parse_time(case["target"])))

    rows = []
    for index, trophy in enumerate(case.get("trophies") or []):
        # `rates` follows the order of `trophies`, so the trophy's
        # position in the list -- not its id -- is its column.
        points = []
        last_rate = None
        for read in history:
            rates = read.get("rates") or []
            if index >= len(rates):
                continue
            rate = rates[index]
            if rate is None:
                continue
            last_rate = rate
            when = read.get("at") or read.get("slot")
            points.append((math.log1p(hours_since_launch(parse_time(when))),
                           rate))
        if last_rate is None:
            q = list(WIDE_PRIOR)
        elif len(history) <= 1:
            q = forecast_trophy_stale(last_rate)
        else:
            q = forecast_trophy_evolving(points, horizon_days, target_x)
        rows.append({"id": trophy["id"], "q": q})
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name)) as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w") as fh:
            json.dump(result, fh)
            fh.write("\n")


if __name__ == "__main__":
    main()
