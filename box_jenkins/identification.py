from statsmodels.tsa import stattools
import numpy as np
import pandas as pd
import altair as alt


class Corr:
    ALPHA = 0.05

    def __init__(self, y, nlags: int = 24) -> None:
        self.y = y
        self.nlags = nlags
        self.est_acf().est_pacf().est_qstat()

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
                "lag": range(0, nlags + 1),
                "values": values,
                "ci_low": confint[:, 0] - values,
                "ci_upp": confint[:, 1] - values,
            }
        )

    def est_acf(self):
        values, confint = stattools.acf(
            x=self.y, nlags=self.nlags, fft=True, alpha=self.ALPHA
        )
        self.acf_df = self.create_corr_df(values, confint, self.nlags)
        return self

    def est_pacf(self):
        values, confint = stattools.pacf(
            x=self.y, nlags=self.nlags, method="ywunbiased", alpha=self.ALPHA
        )
        self.pacf_df = self.create_corr_df(values, confint, self.nlags)
        return self

    def est_qstat(self):
        values, p_vaules = stattools.q_stat(self.y, self.y.shape[0])
        self.qstat_df = pd.DataFrame(
            {"lag": range(1, self.nlags), "values": values, "pvalues": p_vaules}
        )
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
        acf = (
            self.bar(self.acf_df, "ACF")
            + self.plot_ci(self.acf_df, "ci_low")
            + self.plot_ci(self.acf_df, "ci_upp")
        )
        pacf = (
            self.bar(self.pacf_df, "PACF")
            + self.plot_ci(self.pacf_df, "ci_low")
            + self.plot_ci(self.pacf_df, "ci_upp")
        )
        return acf & pacf

    @staticmethod
    def plot_bar(df: pd.DataFrame, name: str) -> alt.vegalite.v4.api.Chart:
        plot = (
            alt.Chart(df[1:])
            .mark_bar()
            .encode(x=alt.X("lag:Q", title="Lag"), y=alt.Y("values:Q", title=name))
            .interactive()
        )
        return plot

    @staticmethod
    def plot_ci(df: pd.DataFrame, type: str) -> alt.vegalite.v4.api.Chart:
        plot = (
            alt.Chart(self.df)
            .mark_area(opacity=0.3, color="red", clip=True)
            .encode(
                x=alt.X("lag:Q", scale=alt.Scale(domain=(1, self.nlags), zero=False)),
                y=f"{type}:Q",
            )
        )
        return plot
