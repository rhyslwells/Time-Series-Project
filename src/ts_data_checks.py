"""
Generic time series data inspection module
- Model-independent checks on a single value column of any polars DataFrame
- Distribution, trend, time-of-day, day-of-week, ACF/PACF, rolling mean/variance,
  stationarity tests (ADF/KPSS), STL decomposition and seasonal/trend strength
- Does not assume any feature engineering (e.g. src/data/generate_metering_features.py
  output) - only a timestamp column and one numeric value column are required
"""

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class DataInspector:
    """Model-independent checks for a single time series column"""

    @staticmethod
    def to_series(df: pl.DataFrame, value_col: str, timestamp_col: str = "timestamp"):
        """Extract (timestamps, values) as numpy arrays, sorted by time"""
        ordered = df.sort(timestamp_col)
        timestamps = ordered[timestamp_col].to_numpy()
        values = ordered[value_col].to_numpy()
        return timestamps, values

    @staticmethod
    def summary_stats(df: pl.DataFrame, value_col: str) -> pl.DataFrame:
        """Distribution and range of the target: count, mean, std, min/max, quartiles"""
        return df.select(
            pl.col(value_col).count().alias("count"),
            pl.col(value_col).null_count().alias("null_count"),
            pl.col(value_col).mean().alias("mean"),
            pl.col(value_col).std().alias("std"),
            pl.col(value_col).min().alias("min"),
            pl.col(value_col).quantile(0.25).alias("p25"),
            pl.col(value_col).median().alias("median"),
            pl.col(value_col).quantile(0.75).alias("p75"),
            pl.col(value_col).max().alias("max"),
        )

    @staticmethod
    def distribution_plot(df: pl.DataFrame, value_col: str, title: str = None) -> go.Figure:
        """Plot: histogram of the target's distribution"""
        values = df[value_col].to_numpy()

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=values, nbinsx=40, marker_color="steelblue"))
        fig.add_vline(
            x=float(np.mean(values)),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {np.mean(values):.3f}",
        )

        fig.update_layout(
            title=title or f"Distribution of {value_col}",
            xaxis_title=value_col,
            yaxis_title="Frequency",
            height=400,
            width=900,
        )
        return fig

    @staticmethod
    def series_plot(
        df: pl.DataFrame,
        value_col: str,
        timestamp_col: str = "timestamp",
        rolling_window: int = 48,
        title: str = None,
    ) -> go.Figure:
        """Plot: raw series over time, with a rolling mean overlay to expose trend/regime changes"""
        timestamps, values = DataInspector.to_series(df, value_col, timestamp_col)
        rolling_mean = (
            df.sort(timestamp_col)
            .select(pl.col(value_col).rolling_mean(window_size=rolling_window))
            .to_series()
            .to_numpy()
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=values,
                mode="lines", name=value_col,
                line=dict(color="steelblue", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=rolling_mean,
                mode="lines", name=f"Rolling mean (window={rolling_window})",
                line=dict(color="red", width=2),
            )
        )

        fig.update_layout(
            title=title or f"{value_col} Over Time",
            xaxis_title="Time",
            yaxis_title=value_col,
            height=450,
            width=1200,
            hovermode="x unified",
        )
        return fig

    @staticmethod
    def time_of_day_plot(
        df: pl.DataFrame,
        value_col: str,
        timestamp_col: str = "timestamp",
        title: str = None,
    ) -> go.Figure:
        """Plot: distribution of the target at each time-of-day slot (boxplot)"""
        slotted = df.with_columns(
            (
                pl.col(timestamp_col).dt.hour().cast(pl.Utf8).str.zfill(2)
                + ":"
                + pl.col(timestamp_col).dt.minute().cast(pl.Utf8).str.zfill(2)
            ).alias("time_of_day")
        ).sort("time_of_day")

        fig = go.Figure()
        fig.add_trace(
            go.Box(
                x=slotted["time_of_day"].to_list(),
                y=slotted[value_col].to_list(),
                marker_color="steelblue",
                showlegend=False,
            )
        )

        fig.update_layout(
            title=title or f"{value_col} by Time of Day",
            xaxis_title="Time of day",
            yaxis_title=value_col,
            height=450,
            width=1200,
        )
        return fig

    @staticmethod
    def day_of_week_plot(
        df: pl.DataFrame,
        value_col: str,
        timestamp_col: str = "timestamp",
        title: str = None,
    ) -> go.Figure:
        """Plot: distribution of the target by day of week (boxplot), Monday first"""
        labeled = df.with_columns(
            pl.col(timestamp_col).dt.weekday().alias("weekday_num")
        )

        fig = go.Figure()
        for weekday_num, name in enumerate(_WEEKDAY_NAMES, start=1):
            day_values = labeled.filter(pl.col("weekday_num") == weekday_num)[value_col]
            if day_values.len() == 0:
                continue
            fig.add_trace(go.Box(y=day_values.to_list(), name=name, marker_color="steelblue", showlegend=False))

        fig.update_layout(
            title=title or f"{value_col} by Day of Week",
            xaxis_title="Day",
            yaxis_title=value_col,
            height=450,
            width=1000,
        )
        return fig

    @staticmethod
    def rolling_stats_plot(
        df: pl.DataFrame,
        value_col: str,
        timestamp_col: str = "timestamp",
        window: int = 48,
        title: str = None,
    ) -> go.Figure:
        """Plot: rolling mean and rolling variance over time (stationarity check)"""
        ordered = df.sort(timestamp_col)
        timestamps = ordered[timestamp_col].to_numpy()
        rolling = ordered.select(
            pl.col(value_col).rolling_mean(window_size=window).alias("rolling_mean"),
            pl.col(value_col).rolling_var(window_size=window).alias("rolling_var"),
        )

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f"Rolling Mean (window={window})", f"Rolling Variance (window={window})"),
        )

        fig.add_trace(
            go.Scatter(x=timestamps, y=rolling["rolling_mean"].to_numpy(), mode="lines", line=dict(color="steelblue")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=timestamps, y=rolling["rolling_var"].to_numpy(), mode="lines", line=dict(color="salmon")),
            row=2, col=1,
        )

        fig.update_yaxes(title_text="Rolling mean", row=1, col=1)
        fig.update_yaxes(title_text="Rolling variance", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_layout(
            title_text=title or f"{value_col}: Rolling Mean & Variance",
            height=600,
            width=1200,
            showlegend=False,
        )
        return fig

    @staticmethod
    def acf_pacf_plot(
        values: np.ndarray,
        nlags: int = 48,
        title: str = "Series",
    ) -> go.Figure:
        """Plot: ACF and PACF of the raw series, each with a 95% white-noise band"""
        from statsmodels.tsa.stattools import acf, pacf

        nlags = min(nlags, len(values) // 2 - 1)
        acf_values = acf(values, nlags=nlags, fft=False)
        pacf_values = pacf(values, nlags=nlags, method="ywm")
        conf_band = 1.96 / np.sqrt(len(values))

        fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))

        fig.add_trace(
            go.Bar(x=list(range(len(acf_values))), y=acf_values, marker_color="steelblue", showlegend=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(x=list(range(len(pacf_values))), y=pacf_values, marker_color="steelblue", showlegend=False),
            row=1, col=2,
        )

        for col in (1, 2):
            fig.add_hline(y=conf_band, line_dash="dash", line_color="red", row=1, col=col)
            fig.add_hline(y=-conf_band, line_dash="dash", line_color="red", row=1, col=col)
            fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, row=1, col=col)

        fig.update_xaxes(title_text="Lag", row=1, col=1)
        fig.update_xaxes(title_text="Lag", row=1, col=2)
        fig.update_yaxes(title_text="ACF", row=1, col=1)
        fig.update_yaxes(title_text="PACF", row=1, col=2)
        fig.update_layout(title_text=f"{title}: Autocorrelation", height=450, width=1200)

        return fig

    @staticmethod
    def stationarity_tests(values: np.ndarray) -> pl.DataFrame:
        """ADF and KPSS stationarity tests, with a combined read.

        ADF's null hypothesis is a unit root (non-stationary); rejecting it (p < 0.05)
        indicates stationarity. KPSS's null hypothesis is stationarity; rejecting it
        (p < 0.05) indicates non-stationarity. The two are complementary: agreement is
        conclusive, disagreement narrows down which kind of non-stationarity is present
        (trend vs difference).
        """
        import warnings
        from statsmodels.tsa.stattools import adfuller, kpss

        adf_stat, adf_pvalue = adfuller(values, autolag="AIC")[:2]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_pvalue = kpss(values, regression="c", nlags="auto")[:2]

        adf_stationary = adf_pvalue < 0.05
        kpss_stationary = kpss_pvalue >= 0.05

        if adf_stationary and kpss_stationary:
            conclusion = "stationary"
        elif not adf_stationary and not kpss_stationary:
            conclusion = "non-stationary"
        elif adf_stationary and not kpss_stationary:
            conclusion = "trend-stationary (detrend before treating as stationary)"
        else:
            conclusion = "difference-stationary (difference before treating as stationary)"

        return pl.DataFrame({
            "test": ["ADF", "KPSS"],
            "statistic": [adf_stat, kpss_stat],
            "p_value": [adf_pvalue, kpss_pvalue],
            "null_hypothesis": ["unit root (non-stationary)", "stationary"],
            "rejects_null_at_0.05": [bool(adf_stationary), bool(not kpss_stationary)],
            "conclusion": [conclusion, conclusion],
        })

    @staticmethod
    def seasonal_decomposition_plot(
        df: pl.DataFrame,
        value_col: str,
        timestamp_col: str = "timestamp",
        period: int = 48,
        title: str = None,
    ) -> go.Figure:
        """Plot: STL decomposition into observed, trend, seasonal, and residual components"""
        from statsmodels.tsa.seasonal import STL

        timestamps, values = DataInspector.to_series(df, value_col, timestamp_col)
        result = STL(values, period=period, robust=True).fit()

        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
        )
        fig.add_trace(go.Scatter(x=timestamps, y=values, mode="lines", line=dict(color="steelblue")), row=1, col=1)
        fig.add_trace(go.Scatter(x=timestamps, y=result.trend, mode="lines", line=dict(color="red")), row=2, col=1)
        fig.add_trace(go.Scatter(x=timestamps, y=result.seasonal, mode="lines", line=dict(color="green")), row=3, col=1)
        fig.add_trace(go.Scatter(x=timestamps, y=result.resid, mode="markers", marker=dict(color="gray", size=3)), row=4, col=1)

        fig.update_layout(
            title_text=title or f"{value_col}: STL Decomposition (period={period})",
            height=800,
            width=1200,
            showlegend=False,
        )
        return fig

    @staticmethod
    def seasonal_strength(
        df: pl.DataFrame,
        value_col: str,
        timestamp_col: str = "timestamp",
        period: int = 48,
    ) -> pl.DataFrame:
        """Hyndman/Wang trend and seasonal strength from an STL decomposition, each in [0, 1].

        strength = max(0, 1 - Var(residual) / Var(component + residual)). Near 0 means the
        component adds nothing beyond noise; near 1 means it accounts for nearly all the
        variation left after the other component is removed.
        """
        from statsmodels.tsa.seasonal import STL

        _, values = DataInspector.to_series(df, value_col, timestamp_col)
        result = STL(values, period=period, robust=True).fit()

        resid_var = np.var(result.resid)
        trend_strength = max(0.0, 1 - resid_var / np.var(result.trend + result.resid))
        seasonal_strength = max(0.0, 1 - resid_var / np.var(result.seasonal + result.resid))

        return pl.DataFrame({
            "period": [period],
            "trend_strength": [trend_strength],
            "seasonal_strength": [seasonal_strength],
        })
