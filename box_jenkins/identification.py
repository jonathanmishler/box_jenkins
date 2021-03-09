from altair.vegalite.v4.schema.channels import Tooltip
from statsmodels.tsa import stattools
import numpy as np
import pandas as pd
import altair as alt


class Corr:
    ALPHA = 0.05

    def __init__(self, y, nlags: int = 24) -> None:
        self.y = y
        self.nlags = nlags
        self.est_acf().est_pacf()

    @staticmethod
    def create_corr_df(
        values: np.ndarray, confint: np.ndarray, nlags: int
    ) -> pd.DataFrame:
        """Creates the correlation dataframe with columns of lags, values, ci_low, and ci_upp.
        The confidence interval lower bound and upper bound is based around a zero value as I like
        this method when plotted better than the CI band being around the estimated value.
        """
        return pd.DataFrame(
            {
                "values": values,
                "ci_low": confint[:, 0] - values,
                "ci_upp": confint[:, 1] - values,
            },
            index=range(0, nlags + 1),
        )

    def est_acf(self):
        values, confint, qstat, pvalues = stattools.acf(
            x=self.y, nlags=self.nlags, fft=True, alpha=self.ALPHA, qstat=True
        )
        self.acf_df = self.create_corr_df(values, confint, self.nlags)
        self.qstat_df = pd.DataFrame(
            {"values": qstat, "pvalues": pvalues}, 
            index=range(1, self.nlags+1)
        )
        return self

    def est_pacf(self):
        values, confint = stattools.pacf(
            x=self.y, nlags=self.nlags, method="ywadjusted", alpha=self.ALPHA
        )
        self.pacf_df = self.create_corr_df(values, confint, self.nlags)
        return self

    @property
    def acf(self):
        return self.acf_df

    @property
    def pacf(self):
        return self.pacf_df

    @property
    def qstat(self):
        return self.qstat_df

    def plot(self):
        acf = self.combined_plot(self.acf_df[1:], "ACF")
        pacf = self.combined_plot(self.pacf_df[1:], "PACF")
        return acf & pacf

    def combined_plot(self, df: pd.DataFrame, name: str):
        df = df.reset_index()
        return (
            self.plot_bar(df, name) + self.plot_ci(df, "ci_low") + self.plot_ci(df, "ci_upp")
        )

    @staticmethod
    def plot_bar(df: pd.DataFrame, name: str) -> alt.vegalite.v4.api.Chart:
        plot = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("index:N", title="Lag"), 
                y=alt.Y("values:Q", title=name),
                tooltip = alt.Tooltip("values", format = ",.3f")
            )
            .properties(width=800, height=150)
        )
        return plot

    @staticmethod
    def plot_ci(df: pd.DataFrame, type: str) -> alt.vegalite.v4.api.Chart:
        plot = (
            alt.Chart(df)
            .mark_bar(opacity=0.3, color="red")
            .encode(
                x=alt.X("index:N"),
                y=f"{type}:Q",
            )
        )
        return plot
