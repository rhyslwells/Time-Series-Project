import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import sys

    sys.path.insert(0, "../")

    import marimo as mo
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import warnings

    from ts_models import SeasonalWindowAverageModel
    from ts_products import (
        exceedance_probability,
        expected_exceedance,
        forecast_quantile,
    )

    warnings.filterwarnings("ignore")
    return (
        ListedColormap,
        SeasonalWindowAverageModel,
        exceedance_probability,
        expected_exceedance,
        forecast_quantile,
        mo,
        np,
        pl,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    A **forecast product** is a signal derived from a forecast that a downstream
    decision can use directly. A raw forecast says "0.8 kWh at 18:00 tomorrow".
    A scheduler, a trader or a grid operator usually wants something else:
    "will the asset be above its limit this evening", "how much energy over the
    day", "how much should I trust this slot".

    The model is held fixed (`SeasonalWindowAverage`, window=7: the simplest model
    that still gives a prediction interval), so the notebook is about what each
    product tells you, and what it cannot. Every product is computed from the
    point forecast and the 80% interval only, with no model-specific code.

    1. **Daily energy, peak, ramp** - the forecast re-expressed in operational quantities
    2. **Uncertainty as an input** - what the interval width says about each slot
    3. **Event probability** - the chance a slot exceeds a threshold, and whether to believe it
    4. **Expected schedule** - turning those probabilities into "high-load windows"
    5. **From probability to decision** - how much is at stake (expected
       exceedance), how cautious to be (a firm planning value), where to draw
       the line (a cost-based cutoff), and a replay of using them as metering
       arrives

    Units: `metering_kwh` is energy per 30-minute slot, so a 5 kW load is 2.5 kWh
    per slot.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 0: Fixed model and base forecast

    10 days train, 4 days test, one EV-charging asset. The model repeats one
    average daily profile (48 half-hour slots) across the test days, with an 80%
    interval around it. That is all the information the products below get.
    """)
    return


@app.cell
def _(SeasonalWindowAverageModel, pl):
    df = pl.read_parquet("../../src/data/metering_data.parquet").sort(
        "asset_id", "timestamp"
    )
    n_test = 4 * 48
    confidence_level = 0.80

    series = df.filter(pl.col("asset_id") == "ASSET_001")
    y = series["metering_kwh"].to_numpy()
    y_train = y[:-n_test]
    y_test = y[-n_test:]
    ts = series["timestamp"].to_numpy()[-n_test:]

    model = SeasonalWindowAverageModel(y_train, season_length=48, window=7)
    model.fit()
    fc = model.forecast(steps=n_test, confidence_level=confidence_level)

    day_labels = [
        ts[i * 48].astype("datetime64[D]").astype(object).strftime("%a %d %b")
        for i in range(4)
    ]
    print(f"test window: {ts[0]} to {ts[-1]}")
    print("test days:", day_labels)
    return confidence_level, day_labels, fc, ts, y_test


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 1: Daily energy, peak and ramp

    $$E_{\\text{day}} = \\sum_t \\hat y_t \\qquad P_{\\text{day}} = \\max_t \\hat y_t \\qquad R_{\\text{day}} = \\max_t |\\hat y_t - \\hat y_{t-1}|$$

    **What are these for?** The forecast is 48 numbers a day. Most operational
    questions are about one number per day, or one number about shape:

    | Product | Question it answers | Who cares |
    |---|---|---|
    | Daily energy $E$ | How much energy will the asset use or export today? | Energy procurement, billing, battery or charger energy budgets |
    | Daily peak $P$ | What is the highest load we should plan capacity for? | Connection limits, demand charges, capacity booking |
    | Max ramp $R$ | How fast does load change between adjacent slots? | Balancing: how quickly generation or flexibility must respond |

    These are simple reshapes of the point forecast, so the real value is the
    comparison with what actually happened: **a derived product has its own
    accuracy, which is not the slot accuracy.** The table puts the forecast
    version next to the actual version for each test day.
    """)
    return


@app.cell
def _(day_labels, fc, np, pl, y_test):
    def day_stats(values):
        days = values.reshape(-1, 48)
        ramps = np.abs(np.diff(days, axis=1)).max(axis=1)
        return days.sum(axis=1), days.max(axis=1), ramps


    e_hat, p_hat, r_hat = day_stats(fc.prediction)
    e_act, p_act, r_act = day_stats(y_test)

    daily = pl.DataFrame(
        {
            "day": day_labels,
            "energy_fcst_kwh": e_hat,
            "energy_actual_kwh": e_act,
            "energy_err_pct": 100 * (e_hat - e_act) / e_act,
            "peak_fcst_kwh": p_hat,
            "peak_actual_kwh": p_act,
            "peak_err_pct": 100 * (p_hat - p_act) / p_act,
            "max_ramp_fcst": r_hat,
            "max_ramp_actual": r_act,
        }
    ).with_columns(pl.exclude("day").round(2))

    slot_mape = 100 * np.mean(
        np.abs(fc.prediction - y_test) / np.maximum(np.abs(y_test), 0.1)
    )
    daily
    return (slot_mape,)


@app.cell
def _(day_labels, fc, plt, ts, y_test):
    fig_1, ax_1 = plt.subplots(figsize=(12, 3.5))
    ax_1.plot(ts, y_test, label="actual", color="C0", alpha=0.8)
    ax_1.plot(ts, fc.prediction, label="forecast", color="C1")
    for _d in range(4):
        _s = slice(_d * 48, (_d + 1) * 48)
        ax_1.hlines(
            fc.prediction[_s].max(), ts[_s][0], ts[_s][-1],
            colors="C1", linestyles="--", linewidth=1,
            label="forecast daily peak" if _d == 0 else None,
        )
        ax_1.hlines(
            y_test[_s].max(), ts[_s][0], ts[_s][-1],
            colors="C0", linestyles=":", linewidth=1.5,
            label="actual daily peak" if _d == 0 else None,
        )
    ax_1.set_ylabel("kWh per 30-min slot")
    ax_1.set_title("Forecast vs actual, with the daily peak each one implies")
    ax_1.legend(ncol=4, loc="upper left")
    fig_1.tight_layout()
    fig_1
    return


@app.cell(hide_code=True)
def _(mo, slot_mape):
    mo.md(f"""
    **Reading the table and plot.** The forecast is the *same* profile every
    day: the 7-day window averages weekdays and weekends together. The test
    window starts on a Saturday, so the two weekend days (lower use) are
    over-forecast by roughly 40%, and the two weekday days (higher use) are
    under-forecast by roughly 10%. A sum does not hide a systematic day-type
    error, it accumulates it.

    Peaks fail differently: the forecast peak is the same each day, so it is
    too high on the quiet days and too low on the busy ones. The ramp columns
    show the same pattern: the forecast's sharpest step is the same every day,
    above what the weekend days did and below what the weekdays did. All three
    products are wrong in the same direction on the same days, because they
    share one cause (the model does not know the day type).

    For comparison the per-slot absolute error is about {slot_mape:.0f}%
    (denominator floored at 0.1 kWh). The point is not that daily numbers are
    better or worse than slot numbers, it is that they fail in a different
    place, so each product's accuracy has to be checked directly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 2: Uncertainty as a decision input

    **What the interval is.** Besides the point forecast, the model returns an
    80% prediction interval: it claims the actual value will fall between
    `lower` and `upper` about 80% of the time. The width

    $$U_t = \\text{upper}_t - \\text{lower}_t$$

    is a statement of **how much to trust that one slot**. Narrow means act on
    the point forecast; wide means it is a rough guess. A downstream optimiser
    can use `lower` as a conservative planning value (assume the load is no
    higher than the 90th percentile, or the generation no lower than the 10th),
    rather than the model deciding for it how cautious to be.

    **The question here is whether the width is believable.** The two plots
    check the width against reality:

    - **Left, interval width over the horizon.** The model assumes error grows
      with distance: each further day adds one more seasonal cycle of error, so
      the width steps up at each day boundary (the model's $\\sqrt{\\text{cycles}}$
      rule). Within a day it is flat.
    - **Right, implied sigma vs actual error by time of day.** The orange line
      is the one standard deviation the model applies to every slot. The blue
      line is how big the forecast error actually was, slot by slot, averaged
      over the four test days. If the model were honest about uncertainty the
      two would roughly agree. They do not: error is near zero overnight (the
      asset is idle, and easy to forecast) and large in the evening, while the
      model uses the same sigma at 03:00 as at 18:00.

    **What to take from this:** the interval is too wide when the asset is idle
    and too narrow when it is busy. But the right-hand blue line averages over
    weekend and weekday days, and most of its evening peak comes from the two
    weekend days, where the model over-forecasts (Section 1): in this run the
    evening RMS error is about 1.1 kWh on the weekend days against about 0.4
    on the weekdays, with the model assuming about 0.27 throughout. So two
    things are wrong, a spread that ignores time of day and a mean that ignores
    day type, and the second dominates here. Anything built on the interval
    inherits both, which matters in the next sections.
    """)
    return


@app.cell
def _(fc, np, plt, y_test):
    errors = (y_test - fc.prediction).reshape(-1, 48)
    sigma_day1 = fc.uncertainty_width[:48].mean() / 2 / 1.2816
    slot_rms = np.sqrt((errors**2).mean(axis=0))

    fig_u, (ax_w, ax_s) = plt.subplots(1, 2, figsize=(12, 3.8))

    ax_w.plot(fc.uncertainty_width)
    for _b in (48, 96, 144):
        ax_w.axvline(_b, color="grey", linestyle=":", linewidth=1)
    ax_w.set_title("Interval width over the 4 test days")
    ax_w.set_xlabel("forecast step (30-min slots from start of test)")
    ax_w.set_ylabel("upper - lower (kWh)")

    hours = np.arange(48) / 2
    ax_s.plot(hours, slot_rms, label="actual error (RMS over 4 days)")
    ax_s.axhline(sigma_day1, color="C1", label="sigma the model assumes (day 1)")
    ax_s.set_title("Assumed uncertainty vs actual error by time of day")
    ax_s.set_xlabel("hour of day")
    ax_s.set_ylabel("kWh")
    ax_s.set_xticks(range(0, 25, 3))
    ax_s.legend()
    fig_u.tight_layout()
    fig_u
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 3: Event probability

    **The idea.** Treat each slot's forecast as a bell curve centred on the point
    forecast, with spread taken from the interval:
    $Y_t \\sim N(\\hat y_t, \\sigma_t^2)$ where $\\sigma_t = U_t / (2z)$ and $z$ is
    set by the confidence level (1.2816 for 80%). Then the probability the slot
    exceeds a threshold $c$ is the area of the curve above $c$:

    $$P(Y_t > c) = 1 - \\Phi\\!\\left(\\frac{c - \\hat y_t}{\\sigma_t}\\right)$$

    It answers "will this asset exceed 5 kW in this slot?" with a probability
    rather than a yes/no on the point forecast, which is the form a
    risk-aware decision wants.

    **It is one probability per slot.** 0.9 at 18:00 means about nine times in
    ten that specific half hour is above 5 kW. Averaging over the whole horizon,
    as the SARIMA notebook's Section 8 does, mixes quiet overnight slots with the
    evening peak and answers no operational question. Meaningful summaries are
    the curve over the day, the **expected number of slots above threshold**
    ($\\sum_t P(Y_t > c)$, valid without independence assumptions), and the
    windows where $P > 0.5$ (next section). "At least one slot above $c$" is
    *not* computable from these, it needs the correlation between slots.

    Also, the SARIMA notebook divides the 80% interval by $2 \\times 1.645$, the
    90% z-value. `ts_products.implied_std` takes $z$ from the confidence level.
    """)
    return


@app.cell
def _(confidence_level, exceedance_probability, fc, np, pl, y_test):
    threshold_kwh = 2.5  # 5 kW over a 30-minute slot
    p_exceed = exceedance_probability(fc, threshold_kwh, confidence_level)
    actual_exceed = y_test > threshold_kwh

    by_slot = (
        pl.DataFrame(
            {
                "slot": np.tile(np.arange(48), len(y_test) // 48),
                "p_exceed": p_exceed,
                "actual_above": actual_exceed.astype(float),
            }
        )
        .group_by("slot")
        .agg(pl.col("p_exceed").mean(), pl.col("actual_above").mean())
        .sort("slot")
    )
    print(f"threshold: {threshold_kwh} kWh/slot (5 kW)")
    print(f"expected slots above: {p_exceed.sum():.1f}   actual: {actual_exceed.sum()}")
    print(f"naive 'mean P over horizon': {p_exceed.mean():.1%}  (not a meaningful number)")
    return actual_exceed, by_slot, p_exceed, threshold_kwh


@app.cell
def _(by_slot, plt):
    fig_p, ax_p = plt.subplots(figsize=(12, 3.5))
    ax_p.plot(by_slot["slot"] / 2, by_slot["p_exceed"], label="forecast P(above 5 kW)")
    ax_p.plot(
        by_slot["slot"] / 2,
        by_slot["actual_above"],
        "o",
        label="fraction of the 4 test days actually above 5 kW",
    )
    ax_p.set_xlabel("hour of day")
    ax_p.set_ylabel("probability / fraction of days")
    ax_p.set_xticks(range(0, 25, 3))
    ax_p.legend()
    fig_p.tight_layout()
    fig_p
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Calibration of the event probability

    A probability is only useful if it means what it says: events forecast at 30%
    should happen about 30% of the time. The table checks this. Every one of the
    192 test slots has a forecast probability; they are grouped into bands, and
    for each band:

    - `n_slots` - how many slots fell in the band
    - `mean_forecast_p` - the average probability the model gave them
    - `observed_freq` - the fraction that actually exceeded the threshold

    **Reading it:** a well-calibrated model has `observed_freq` close to
    `mean_forecast_p` in every row. If `observed_freq` is *lower* in the high
    bands, the model is over-confident (it says "almost certain" about events
    that do not always happen); if *higher* in the low bands, it is
    over-confident the other way (it says "very unlikely" about events that
    occur). The plot shows the same thing: points on the diagonal are
    calibrated, below the diagonal is over-confident.
    """)
    return


@app.cell
def _(actual_exceed, np, p_exceed, pl):
    edges = np.array([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0001])
    band_labels = ["0-5%", "5-25%", "25-50%", "50-75%", "75-95%", "95-100%"]
    bins = np.digitize(p_exceed, edges) - 1
    calibration = (
        pl.DataFrame(
            {"bin": bins, "p": p_exceed, "hit": actual_exceed.astype(float)}
        )
        .group_by("bin")
        .agg(
            pl.len().alias("n_slots"),
            pl.col("p").mean().alias("mean_forecast_p"),
            pl.col("hit").mean().alias("observed_freq"),
        )
        .sort("bin")
        .with_columns(
            pl.col("bin").map_elements(lambda b: band_labels[b], return_dtype=pl.String).alias("forecast_band")
        )
        .drop("bin")
        .select("forecast_band", "n_slots", "mean_forecast_p", "observed_freq")
        .with_columns(pl.col("mean_forecast_p", "observed_freq").round(3))
    )
    calibration
    return (calibration,)


@app.cell
def _(calibration, plt):
    fig_c, ax_c = plt.subplots(figsize=(4.5, 4.5))
    ax_c.plot([0, 1], [0, 1], color="grey", linestyle="--", label="perfect calibration")
    ax_c.scatter(
        calibration["mean_forecast_p"],
        calibration["observed_freq"],
        s=calibration["n_slots"] * 4,
        color="C1",
        label="band (size = n_slots)",
    )
    ax_c.set_xlabel("mean forecast probability")
    ax_c.set_ylabel("observed frequency")
    ax_c.set_xlim(-0.02, 1.02)
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.legend(loc="upper left")
    fig_c.tight_layout()
    fig_c
    return


@app.cell(hide_code=True)
def _(calibration, mo):
    top = calibration.tail(1)
    mo.md(f"""
    **What this run shows.** The mass of slots sits at the two extremes: most
    are forecast as essentially never above 5 kW (overnight) or essentially
    always (evening). In the top band the model's mean probability is
    {top["mean_forecast_p"][0]:.0%} but only {top["observed_freq"][0]:.0%} of
    those slots actually exceeded the threshold. The shortfall sits entirely on
    the two weekend days: there the evening slots were forecast as near-certain
    exceedances and often did not happen, while on the two weekdays every
    top-band slot did exceed. So the over-confidence is mainly the day-type
    bias from Section 1, made worse by a spread (Section 2) too narrow to admit
    it. The middle bands hold few slots, so their frequencies are noisy and
    should not be over-read.

    **Practical consequence:** use the probabilities for ranking and for
    rough planning, but do not treat a "99%" as 99%. A model that knows the day
    type, with time-of-day-aware uncertainty (e.g. quantile regression), is
    what would fix this.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 4: Expected schedule

    **The idea.** Apply the event rule across the horizon: a slot is an
    *expected high-load slot* if $P(Y_t > 5\\,\\text{kW}) > 0.5$. Consecutive
    flagged slots form windows, which turns the forecast into "expect the
    asset to be charging 17:00-21:00". That is the form a scheduler consumes,
    for example to avoid starting a flexible job in that window.

    **How to read the plot.** Each panel is a calendar: one row per test day
    (the date is on the y-axis), and the x-axis is the hour of that day. A
    dark cell means "above 5 kW in that half hour".

    - **Forecast** - slots the model expects above 5 kW. The rows are
      identical because the model repeats one daily profile.
    - **Actual** - slots that were really above 5 kW. Rows differ day to day.
    - **Agreement** - the two compared: green = hit (flagged and happened),
      orange = false alarm (flagged, did not happen), red = missed (not
      flagged, but it happened), white = correctly quiet.

    **What you are meant to infer:** whether the expected window lines up with
    when load really occurred. Look for the window being right in the evening
    but wrong at the edges, and for days (here the weekend) where it flags a
    window that never materialised. The count of each category is printed
    below.
    """)
    return


@app.cell
def _(ListedColormap, actual_exceed, day_labels, np, p_exceed, plt):
    expected_high = p_exceed > 0.5
    f_grid = expected_high.reshape(4, 48)
    a_grid = actual_exceed.reshape(4, 48)

    agree = np.zeros((4, 48))
    agree[f_grid & a_grid] = 1  # hit
    agree[f_grid & ~a_grid] = 2  # false alarm
    agree[~f_grid & a_grid] = 3  # missed

    extent = [0, 24, 4, 0]
    fig_s, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for _ax, _grid, _cmap, _title in [
        (axes[0], f_grid, "Blues", "Forecast: expected above 5 kW (P > 0.5)"),
        (axes[1], a_grid, "Blues", "Actual: above 5 kW"),
        (
            axes[2],
            agree,
            ListedColormap(["white", "#4caf50", "#ff9800", "#e53935"]),
            "Agreement: green = hit, orange = false alarm, red = missed",
        ),
    ]:
        _ax.imshow(_grid, aspect="auto", cmap=_cmap, extent=extent, interpolation="nearest", vmin=0)
        _ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        _ax.set_yticklabels(day_labels)
        _ax.set_title(_title, loc="left", fontsize=10)
    axes[2].set_xticks(range(0, 25, 2))
    axes[2].set_xlabel("hour of day")
    fig_s.tight_layout()
    fig_s
    return (expected_high,)


@app.cell
def _(actual_exceed, expected_high):
    hit = int((expected_high & actual_exceed).sum())
    false_alarm = int((expected_high & ~actual_exceed).sum())
    missed = int((~expected_high & actual_exceed).sum())
    print(f"hit {hit}  false alarm {false_alarm}  missed {missed}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Future idea: live schedule tracking (not built)

    The agreement plot above is retrospective: the whole day is already known.
    In practice the same classification would run as actuals arrive, which
    suggests a small interactive app (Dash, or a marimo app with a refresh) for a
    chosen asset:

    - The user picks the **threshold** (the expected value, e.g. 5 kW) and the
      asset.
    - The day opens as the forecast schedule only: every slot blue where the
      model expects exceedance, blank elsewhere.
    - As each half hour completes, the live metering reading is compared with the
      threshold and that slot is coloured **green** (expected and happened),
      **orange** (expected, did not happen) or **red** (not expected, but it
      happened), with a marker for the current time.
    - A running tally of hits, false alarms and misses for the day, and the
      expected-versus-actual slot count so far, would show immediately whether
      the forecast is holding up or the day is departing from it.

    This is the forecast product used as a monitor rather than a plan, and it
    connects to the residual-anomaly idea on the
    [theory page](../theory/forecast_products.md): a run of red or orange slots
    is the early sign that the asset, or the model, has changed. It needs a
    streaming or periodically refreshed source of actuals, which the synthetic
    dataset does not provide, so a first version could replay the test days one
    slot at a time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 5: From probability to decision

    Sections 3 and 4 say *how likely* a slot is to exceed 5 kW. A decision needs
    three more things: **how much is at stake**, **how cautious to be**, and
    **where to draw the line** between acting and not acting. One product for
    each, using the same fixed model and the same 5 kW threshold. A replay at
    the end shows them being used as metering arrives.

    ### 5a. Expected exceedance: how much, not just whether

    Two slots can both have a 60% chance of exceeding 5 kW, one by a whisker and
    one by a wide margin. If the cost is a capacity charge or a penalty, the
    *amount* over the limit is what matters:

    $$E[(Y_t - c)^+] = \\sigma_t\\,[\\varphi(d_t) + d_t\\,\\Phi(d_t)], \\qquad d_t = (\\hat y_t - c)/\\sigma_t$$

    where $\\varphi$ and $\\Phi$ are the normal density and distribution. The
    result is in kWh over the limit, per slot. Summed over a day it is the
    expected energy above the limit.
    """)
    return


@app.cell
def _(
    day_labels,
    actual_exceed,
    confidence_level,
    expected_exceedance,
    fc,
    p_exceed,
    pl,
    threshold_kwh,
    y_test,
):
    exp_over = expected_exceedance(fc, threshold_kwh, confidence_level)
    act_over = (y_test - threshold_kwh).clip(min=0)

    over_daily = pl.DataFrame(
        {
            "day": day_labels,
            "slots_expected": p_exceed.reshape(4, 48).sum(axis=1),
            "slots_actual": actual_exceed.reshape(4, 48).sum(axis=1).astype(float),
            "over_expected_kwh": exp_over.reshape(4, 48).sum(axis=1),
            "over_actual_kwh": act_over.reshape(4, 48).sum(axis=1),
        }
    ).with_columns(pl.exclude("day").round(2))
    over_daily
    return act_over, exp_over


@app.cell
def _(act_over, exp_over, np, plt):
    _hours = np.arange(48) / 2
    fig_o, ax_o = plt.subplots(figsize=(12, 3.5))
    ax_o.bar(_hours - 0.1, exp_over.reshape(4, 48).mean(axis=0), width=0.2, label="expected overshoot (forecast)")
    ax_o.bar(_hours + 0.1, act_over.reshape(4, 48).mean(axis=0), width=0.2, label="actual overshoot")
    ax_o.set_xlabel("hour of day")
    ax_o.set_ylabel("kWh above 5 kW, mean over 4 days")
    ax_o.set_xticks(range(0, 25, 3))
    ax_o.legend()
    fig_o.tight_layout()
    fig_o
    return


@app.cell(hide_code=True)
def _(act_over, exp_over, mo):
    mo.md(f"""
    **Reading it.** The table compares, per day, the expected number of slots
    over the limit and the expected energy over it with what happened. Over the
    four days the forecast expected {exp_over.sum():.1f} kWh over the limit and
    {act_over.sum():.1f} kWh occurred. The bars show where in the day that
    exposure sits: mainly a morning block (about 06:00-08:30) and a larger
    evening block (about 16:00-21:00), which is where a mitigation (delaying
    a charge, drawing on a battery) would need to act. The daily columns show
    the Section 1 problem again: the same expected exposure every day, but
    the weekend days saw far less than expected and the weekdays somewhat less
    than their share. Quote this as a planning estimate with that known bias.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 5b. A firm planning value: how cautious to be

    The point forecast is the middle of the range. A commitment (a capacity
    offered to a market, a load you promise will be there to curtail) should be
    a value you will *meet nearly all the time*. Taking the 10th percentile of
    the forecast distribution gives a load level that should be reached or
    exceeded 90% of the time:

    $$\\text{firm}_t = \\hat y_t + z_{0.10}\\,\\sigma_t \\quad(\\text{floored at } 0)$$

    The check is simple: how often was the actual at or above the firm value?
    The target is 90%. The right plot extends the check to every percentile: the
    nominal level on the x-axis, how often the actual fell *below* that
    percentile on the y-axis. On the diagonal the distribution is honest; the
    flat sigma from Section 2 shows up as a departure from it.
    """)
    return


@app.cell
def _(confidence_level, fc, forecast_quantile, np, pl, plt, ts, y_test):
    firm_q = 0.10
    firm = np.maximum(forecast_quantile(fc, firm_q, confidence_level), 0)
    met = y_test >= firm
    busy = fc.prediction > 1.0
    met_all, met_busy = met.mean(), met[busy].mean()

    firm_summary = pl.DataFrame(
        {
            "slots": ["all", "busy (forecast > 1 kWh)"],
            "n_slots": [len(met), int(busy.sum())],
            "firm_met_fraction": [met_all, met_busy],
            "target": [1 - firm_q, 1 - firm_q],
        }
    ).with_columns(pl.col("firm_met_fraction").round(3))

    _qs = np.linspace(0.05, 0.95, 19)
    _obs_all = [(y_test <= forecast_quantile(fc, q, confidence_level)).mean() for q in _qs]
    _obs_busy = [
        (y_test[busy] <= forecast_quantile(fc, q, confidence_level)[busy]).mean() for q in _qs
    ]

    fig_f, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    ax_t.plot(ts, y_test, label="actual", color="C0", alpha=0.8)
    ax_t.plot(ts, fc.prediction, label="forecast", color="C1")
    ax_t.plot(ts, firm, label="firm value (P10)", color="C2")
    _breach = y_test < firm
    ax_t.scatter(ts[_breach], y_test[_breach], color="red", s=14, zorder=3, label="below firm value")
    ax_t.set_ylabel("kWh per 30-min slot")
    ax_t.legend(ncol=4, loc="upper left")
    ax_r.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax_r.plot(_qs, _obs_all, "o-", label="all slots")
    ax_r.plot(_qs, _obs_busy, "o-", label="busy slots")
    ax_r.set_xlabel("nominal percentile")
    ax_r.set_ylabel("fraction of actuals below it")
    ax_r.legend()
    fig_f.tight_layout()
    return fig_f, firm_summary, met_all, met_busy


@app.cell
def _(fig_f):
    fig_f
    return


@app.cell
def _(firm_summary):
    firm_summary
    return


@app.cell(hide_code=True)
def _(met_all, met_busy, mo):
    mo.md(f"""
    **Reading it.** The firm value was meant to be met 90% of the time. Across
    all slots it was met {met_all:.0%}, and in the busy slots, which are the
    ones that carry a commitment, only {met_busy:.0%}. All of the red points on
    the left plot are on the two weekend days, where the model over-forecasts
    (Section 1); on the weekdays the firm value was never breached. A
    commitment sized from this interval would under-deliver when it matters. The
    fix is a better distribution (day-type and time-of-day-aware), or a more
    cautious percentile picked by running the check on the right-hand plot
    against history until the observed fraction matches the target.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 5c. A cost-based cutoff: where to draw the line

    Section 4 flagged a slot when $P > 0.5$. That is only right if a false alarm
    and a miss cost the same. Usually they do not. Say acting unnecessarily
    (shifting a load that was not going to breach) costs 1 unit and missing a
    real breach costs 4. Acting is worth it whenever the expected cost of not
    acting exceeds the cost of acting:

    $$P \\cdot c_{\\text{miss}} > (1 - P)\\cdot c_{\\text{false alarm}} \\;\\Rightarrow\\; P > \\frac{c_{\\text{fa}}}{c_{\\text{fa}} + c_{\\text{miss}}}$$

    so the cutoff here should be 0.2, not 0.5. The plot sweeps the cutoff and
    totals the cost on the test slots, to see how that theory holds up against
    the data. With a perfectly calibrated forecast the two would agree.
    """)
    return


@app.cell
def _(actual_exceed, np, p_exceed, pl, plt):
    cost_fa, cost_miss = 1.0, 4.0
    p_star = cost_fa / (cost_fa + cost_miss)


    def cost_at(cutoff):
        flagged = p_exceed > cutoff
        fa = int((flagged & ~actual_exceed).sum())
        ms = int((~flagged & actual_exceed).sum())
        return cost_fa * fa + cost_miss * ms, fa, ms


    _cutoffs = np.linspace(0, 0.99, 100)
    _costs = np.array([cost_at(c)[0] for c in _cutoffs])
    best_cutoff = float(_cutoffs[_costs.argmin()])

    cutoff_table = pl.DataFrame(
        {
            "cutoff": ["0.5 (Section 4)", f"{p_star:.2f} (theory)", f"{best_cutoff:.2f} (best on test)"],
            "false_alarms": [cost_at(c)[1] for c in (0.5, p_star, best_cutoff)],
            "missed": [cost_at(c)[2] for c in (0.5, p_star, best_cutoff)],
            "total_cost": [cost_at(c)[0] for c in (0.5, p_star, best_cutoff)],
        }
    )

    fig_k, ax_k = plt.subplots(figsize=(12, 3.2))
    ax_k.plot(_cutoffs, _costs)
    ax_k.axvline(0.5, color="grey", linestyle=":", label="0.5")
    ax_k.axvline(p_star, color="C1", linestyle="--", label="theory (0.2)")
    ax_k.axvline(best_cutoff, color="C2", linestyle="--", label="best on test")
    ax_k.set_xlabel("probability cutoff for flagging a slot")
    ax_k.set_ylabel("total cost (4 days)")
    ax_k.legend()
    fig_k.tight_layout()
    return cutoff_table, fig_k, p_star


@app.cell
def _(fig_k):
    fig_k
    return


@app.cell
def _(cutoff_table):
    cutoff_table
    return


@app.cell(hide_code=True)
def _(cutoff_table, mo):
    _half, _theory = cutoff_table.row(0, named=True), cutoff_table.row(1, named=True)
    mo.md(f"""
    **Reading it.** Lowering the cutoff trades more false alarms for fewer
    misses. Moving from 0.5 to the theory cutoff takes false alarms from
    {_half["false_alarms"]} to {_theory["false_alarms"]} and misses from
    {_half["missed"]} to {_theory["missed"]}, and total cost from
    {_half["total_cost"]} to {_theory["total_cost"]}, because a miss is four
    times as expensive. Going lower than the theory value buys little more.
    Treat the "best on test" cutoff as noisy: it is fitted to 192 slots, four
    days, and would move with a different window. The principled choice is the
    cost ratio; the data is the check.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 5d. In practice: replaying a day as metering arrives

    In operation the day-ahead forecast is fixed at the start of the day and
    actuals arrive every half hour. Each time a reading lands the question is:
    *given what has happened so far, what should I do about the rest of the day?*
    The replay below plays back one test day. Move the slider to change
    "now" (interactive when the notebook is run locally, the published page
    shows the default).

    - **Realised slots** are coloured with the 5c cutoff: green = flagged and
      happened, orange = flagged but did not, red = not flagged but it
      happened, grey = correctly quiet.
    - **The rest of the day** shows the forecast with its 80% band, and the
      orange shading marks upcoming slots whose probability is above the
      cutoff: these are the slots to act on.
    - The text underneath keeps a running score, and compares the overshoot
      so far with what was expected. If reality is running well above
      expectation by mid-afternoon, the evening is probably worse than the
      forecast says.
    """)
    return


@app.cell
def _(day_labels, mo):
    day_pick = mo.ui.dropdown(day_labels, value=day_labels[2], label="day")
    now_slider = mo.ui.slider(
        0, 47, value=34, step=1, full_width=True,
        label="latest slot received (0 = 00:00-00:30, 34 = 17:00-17:30)",
    )
    mo.hstack([day_pick, now_slider], justify="start")
    return day_pick, now_slider


@app.cell
def _(
    actual_exceed,
    day_labels,
    day_pick,
    exp_over,
    fc,
    act_over,
    mo,
    now_slider,
    np,
    p_exceed,
    p_star,
    plt,
    threshold_kwh,
    y_test,
):
    _d = day_labels.index(day_pick.value)
    _now = now_slider.value
    _sl = slice(_d * 48, (_d + 1) * 48)
    _hours = np.arange(48) / 2
    _done = np.arange(48) <= _now
    _y, _pred = y_test[_sl], fc.prediction[_sl]
    _lo, _hi = fc.lower[_sl], fc.upper[_sl]
    _p = p_exceed[_sl]
    _flag = _p > p_star
    _act = actual_exceed[_sl]

    fig_l, ax_l = plt.subplots(figsize=(12, 3.8))
    ax_l.plot(_hours, _pred, color="C1", linestyle="--", label="forecast")
    ax_l.fill_between(_hours[~_done], _lo[~_done], _hi[~_done], color="C1", alpha=0.2, label="80% band (to come)")
    ax_l.axhline(threshold_kwh, color="grey", linestyle=":", label="5 kW")
    for _i in np.where(~_done & _flag)[0]:
        ax_l.axvspan(_hours[_i], _hours[_i] + 0.5, color="orange", alpha=0.25)
    _colour = np.where(
        _flag & _act, "#4caf50",
        np.where(_flag & ~_act, "#ff9800", np.where(~_flag & _act, "#e53935", "#9e9e9e")),
    )
    ax_l.scatter(_hours[_done], _y[_done], c=_colour[_done], s=30, zorder=3, label="received")
    ax_l.axvline((_now + 1) / 2, color="black", linewidth=1)
    ax_l.set_xlabel("hour of day")
    ax_l.set_ylabel("kWh per 30-min slot")
    ax_l.set_xticks(range(0, 25, 2))
    ax_l.legend(ncol=4, loc="upper left")
    fig_l.tight_layout()

    _hit = int((_flag & _act & _done).sum())
    _fa = int((_flag & ~_act & _done).sum())
    _ms = int((~_flag & _act & _done).sum())
    _over_real = act_over[_sl][_done].sum()
    _over_exp = exp_over[_sl][_done].sum()
    _ahead = np.where(~_done & _flag)[0]
    _rest_exp = exp_over[_sl][~_done].sum()
    _clock = f"{(_now + 1) // 2:02d}:{30 * ((_now + 1) % 2):02d}"
    _next = (
        f"first alerted slot ahead: {_ahead[0] / 2:.1f}h (P = {_p[_ahead[0]]:.0%}), {len(_ahead)} alerted slots in all"
        if len(_ahead)
        else "no upcoming slot above the cutoff"
    )

    mo.vstack(
        [
            fig_l,
            mo.md(
                f"""
    **{day_pick.value}, readings received up to {_clock}.**

    - **So far:** {_hit} hits, {_fa} false alarms, {_ms} missed.
      Overshoot above 5 kW: {_over_real:.2f} kWh realised against
      {_over_exp:.2f} kWh expected for the same slots.
    - **Ahead:** {_next}. Expected overshoot still to come:
      {_rest_exp:.2f} kWh.
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **What this is and is not.** The forecast does not change during the day:
    the replay only compares incoming readings with a fixed plan. That is
    already useful (a running score, a clear list of slots to act on, an early
    warning when realised overshoot departs from expected), but the next step is
    to let the readings update the forecast itself, for example by correcting
    the remaining slots for the error seen so far (an intraday update). The
    fixed model here cannot do that. See also the live app idea at the end of
    Section 4.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Findings

    - Every product inherits the model's systematic errors. The weekday/weekend
      blindness of a 7-day seasonal average shows up in daily energy, in peaks,
      and in the schedule.
    - Derived products have their own accuracy. Daily energy, peak and ramp need
      to be checked against actuals directly, not assumed from slot metrics.
    - The interval is only useful if its width is believable. Here it is flat
      across the day while real error is not (too wide overnight, too narrow in
      the evening), and the larger problem is the mean's day-type bias.
    - Event probability is per slot. Averaging it over a horizon is not
      meaningful; expected exceedance count and windows are.
    - Event probabilities are only as good as the forecast beneath them: slots
      forecast at 95%+ were observed well below 95%, entirely on the weekend
      days. Use them to rank, not to quote.
    - A probability becomes a decision with three more inputs: how much is at
      stake (expected exceedance), how cautious to be (a firm percentile,
      checked against history), and a cutoff set by the cost of a false alarm
      against a miss rather than a default 0.5. The firm P10 value was met far
      less than the 90% intended, because of the same weekend bias.
    - In live use the forecast is fixed and readings arrive each half hour, so
      the products act as a running score and a list of slots to act on. The
      next step is letting the readings update the forecast.

    ### Not covered here

    Baseline and response estimation, residual anomaly detection, portfolio
    aggregation, flexibility ranges and scenario generation. See the
    [forecast products theory page](../theory/forecast_products.md).
    """)
    return


if __name__ == "__main__":
    app.run()
