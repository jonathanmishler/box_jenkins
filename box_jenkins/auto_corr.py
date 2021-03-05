from statsmodels.tsa import stattools
import numpy as np
import pandas as pd
import altair as alt


class AutoCorr:
    def __init__(self, x, nlags: int = 12) -> None:
        pass


class ACF:
    NAME = "ACF"

    def __init__(self, y, nlags: int = 12) -> None:
        self._nlags = nlags + 1
        self.values, self.confint, self.qstat, self.qstat_pvalues = stattools.acf(
            y, nlags=self._nlags, fft=True, alpha=0.05, qstat=True
        )
        self.df = pd.DataFrame(
            {
                "lag": range(0, self._nlags + 1),
                "values": self.values,
                "ci_low": self.confint[:, 0] - self.values,
                "ci_upp": self.confint[:, 1] - self.values,
                "qstat": np.insert(self.qstat, 0, 0),
                "pvalues": np.insert(self.qstat_pvalues, 0, 0),
            }
        )

    @property
    def nlags(self):
        return self._nlags - 1

    def plot(self):
        bar = (
            alt.Chart(self.df[1 : self._nlags])
            .mark_bar()
            .encode(x=alt.X("lag:Q", title="Lag"), y=alt.Y("values:Q", title=self.NAME))
        )

        return (bar + self.plot_ci("ci_low") + self.plot_ci("ci_upp"))

    def plot_ci(self, type: str):
        plot = (
            alt.Chart(self.df)
            .mark_area(opacity=0.3, color="red", clip=True)
            .encode(
                x=alt.X("lag:Q", scale=alt.Scale(domain=(1, self.nlags), zero=False)),
                y=f"{type}:Q",
            )
        )
        return plot


class PACF:
    pass