import matplotlib.pyplot as plt
import pandas as pd

from matcher import Matcher
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from scipy.stats import mstats
from typing import (
    Dict,
    Optional,
    Tuple,
    List
) 


class MonthlyClimateIndexMatcher(Matcher):
    climate_index: pd.DataFrame


    def __init__(self) -> None:
        super().__init__()


    def boxplot(
            self,
            index='PDSI', 
            prev: bool = False,
            month: Optional[str] = None,
            classes: Optional[List] = None,
            ylims: Optional[List] = None
        ) -> Tuple[Optional[Figure], Axes]:

        r"""
        Params:
            index: PDSI, Area, SPEI
            prev: 
            month: 
            classes: 
            ylims: 
        """

        self.__validate_inedx__(index)
        classes = classes if classes else self.__get_classes__()
        groups = self.__get_classes_rows__(self.climate_index)

        if prev:
            df = self.__get_shifted_df__(self.climate_index)
        else:
            df = self.climate_index

        column = month if month else index

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), dpi=200)
        ax.boxplot([list(df.loc[groups[j]][column]) for j in classes])
        title = f"{index} {month if month else ''}{' prev' if prev else ''}"
        ax.set_title(title)
        ax.set_xticklabels([cl + 1 for cl in classes])
        ax.set_xlabel('Class')

        if ylims:
            ax.set_ylim(ylims)

        return fig, ax


    def kruskal_wallis_test(
            self,
            index='PDSI',
            prev: bool = False,
            month: Optional[str] = None,
            classes: Optional[List] = None
        ) -> Tuple[float, float]:

        r"""
        Params:
            index: PDSI, Area, SPEI
            prev: 
            month:
            classes:
        """

        self.__validate_inedx__(index)
        classes = classes if classes else self.__get_classes__()
        groups = self.__get_classes_rows__(self.climate_index)

        if prev:
            df = self.__get_shifted_df__(self.climate_index)
        else:
            df = self.climate_index

        column = month if month else index

        s, p = mstats.kruskalwallis(
                *[list(df.loc[groups[j], column]) for j in classes]
            )

        return s, p
    

    @staticmethod
    def __get_shifted_df__(df) -> pd.DataFrame:
        df_copy = df.copy()
        columns = [col for col in df_copy.columns if col not in ['Year', 'Class']]
        for column in columns:
            df_copy[column] = df_copy[column].shift(1)
        return df_copy.dropna()

