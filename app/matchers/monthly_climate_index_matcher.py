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
    name: str
    climate_index: pd.DataFrame


    def __init__(self, name: str, path: str) -> None:
        self.name = name
        self.climate_index = pd.read_csv(path)


    def boxplot(
            self,
            classes_df: pd.DataFrame,
            prev: bool = False,
            month: Optional[str] = None,
            classes: Optional[List] = None,
            ylims: Optional[List] = None
        ) -> Tuple[Optional[Figure], Axes]:

        r"""
        Params:
            classes_df: Dataframe with 'Class' and 'Year' columns.
            prev: Флаг того, сравнивается ли климатика этого года или предыдущего
            month: Месяц, по которому сравниваем. По-умолчанию сравниваем по Mean.
            classes: Классы, для которых происходит сравнение. По-умолчанию: все.
            ylims: Пределы по оси Y.
        """

        classes = classes if classes else set(classes_df['Class'])
        groups = self.__get_classes_rows__(self.climate_index)

        df = self.__merge_with_classes_(self.climate_index, classes_df)

        if prev:
            df = self.__get_shifted_df__(df)

        column = month if month else 'Mean'

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), dpi=200)
        ax.boxplot([list(df.loc[groups[j]][column]) for j in classes])
        title = f"{self.name} {month if month else ''}{' prev' if prev else ''}"
        ax.set_title(title)
        ax.set_xticklabels([cl + 1 for cl in classes])
        ax.set_xlabel('Class')

        if ylims:
            ax.set_ylim(ylims)

        return fig, ax


    def kruskal_wallis_test(
            self,
            classes_df: pd.DataFrame,
            prev: bool = False,
            month: Optional[str] = None,
            classes: Optional[List] = None
        ) -> Tuple[float, float]:

        r"""
        Params:
            classes_df: Dataframe with 'Class' and 'Year' columns
            prev: Флаг того, сравнивается ли климатика этого года или предыдущего
            month: Месяц, по которому сравниваем. По-умолчанию сравниваем по Mean.
            classes: Классы, для которых происходит сравнение. По-умолчанию: все.
        """

        classes = classes if classes else set(classes_df['Class'])
        groups = self.__get_classes_rows__(self.climate_index)

        df = self.__merge_with_classes_(self.climate_index, classes_df)

        if prev:
            df = self.__get_shifted_df__(df)

        column = month if month else 'Mean'

        s, p = mstats.kruskalwallis(
                *[list(df.loc[groups[j], column]) for j in classes]
            )

        return s, p
    
    
    @staticmethod
    def __merge_with_classes_(df:pd.DataFrame, classes_df: pd.DataFrame) -> pd.DataFrame:
        result = df.merge(classes_df[['Year', 'Class']], on='Year', how='left')
        return result


    @staticmethod
    def __get_shifted_df__(df) -> pd.DataFrame:
        df_copy = df.copy()
        columns = [col for col in df_copy.columns if col not in ['Year', 'Class']]
        for column in columns:
            df_copy[column] = df_copy[column].shift(1)
        return df_copy.dropna()
