import matplotlib.pyplot as plt
import pandas as pd

from app.matchers.matcher import Matcher
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from scipy.stats import mstats
from typing import (
    Dict,
    Optional,
    Tuple,
    List
)

from app.climate_indexes.climate_index import ClimateIndex


class MonthlyClimateIndexMatcher(Matcher):

    def get_values_per_class(
            self,
            climate_index: ClimateIndex,
            classes_df: pd.DataFrame,
            prev: bool = False,
            months: Optional[List[str]] = None,
            classes: Optional[List] = None
    ) -> Dict[int, List]:

        r"""
        Получает все значения индексов по классам
        Params:
            classes_df: Dataframe with 'Class' and 'Year' columns.
            prev: Флаг того, сравнивается ли климатика этого года или предыдущего
            months: Месяцы, по среднему которых сравниваем. По-умолчанию сравниваем по среднему за год.
            classes: Классы, для которых происходит сравнение. По-умолчанию: все.
        
        Возвращает словарь, где в ключах номер класса, а в значениях, все значения климатического индекса для этого класса
        """

        classes = classes if classes else set(classes_df['Class'])

        df = self.__merge_with_classes__(climate_index.data, classes_df)
        groups = self.__get_classes_rows__(df)

        if prev:
            df = self.__get_shifted_df__(df)

        columns = months if months else df.drop(columns=['Year']).columns
        result = {j: self.__filter_nan__(list(df.loc[groups[j]][columns].mean(axis=1))) for j in classes}

        return result

    def boxplot(
            self,
            climate_index: ClimateIndex,
            classes_df: pd.DataFrame,
            prev: bool = False,
            months: Optional[List[str]] = None,
            classes: Optional[List] = None,
            ylims: Optional[List[float]] = None
    ) -> Tuple[Figure, Axes]:

        r"""
        Params:
        Строит boxplot индекса по классам
            classes_df: Dataframe with 'Class' and 'Year' columns.
            prev: Флаг того, сравнивается ли климатика этого года или предыдущего
            months: Месяцы, по среднему которых сравниваем. По-умолчанию сравниваем по среднему за год.
            classes: Классы, для которых происходит сравнение. По-умолчанию: все.
            ylims: Пределы по оси Y.
        """

        classes = classes if classes else set(classes_df['Class'])

        values_per_class = self.get_values_per_class(
            climate_index,
            classes_df,
            prev,
            months,
            classes
        )

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=200)
        ax.boxplot(values_per_class.values())
        title = f"{climate_index.name} {' prev' if prev else ''}"
        ax.set_title(title)
        ax.set_xticklabels([cl + 1 for cl in classes])
        ax.set_xlabel('Class')

        if ylims:
            ax.set_ylim(ylims)

        return fig, ax

    def kruskal_wallis_test(
            self,
            climate_index: ClimateIndex,
            classes_df: pd.DataFrame,
            prev: bool = False,
            months: Optional[List[str]] = None,
            classes: Optional[List] = None
    ) -> Tuple[float, float]:

        r"""
        Params:
            classes_df: Dataframe with 'Class' and 'Year' columns
            prev: Флаг того, сравнивается ли климатика этого года или предыдущего
            months: Месяцы, по среднему которых сравниваем. По-умолчанию сравниваем по среднему за год.
            classes: Классы, для которых происходит сравнение. По-умолчанию: все.
        """

        classes = classes if classes else set(classes_df['Class'])

        values_per_class = self.get_values_per_class(
            climate_index,
            classes_df,
            prev,
            months,
            classes
        )

        s, p = mstats.kruskalwallis(
            *values_per_class.values()
        )

        return s, p

    @staticmethod
    def __merge_with_classes__(df: pd.DataFrame, classes_df: pd.DataFrame) -> pd.DataFrame:
        result = df.merge(classes_df[['Year', 'Class']], on='Year', how='left')
        return result.reset_index(drop=True)

    @staticmethod
    def __get_shifted_df__(df) -> pd.DataFrame:
        df_copy = df.copy()
        columns = [col for col in df_copy.columns if col not in ['Year', 'Class']]
        for column in columns:
            df_copy[column] = df_copy[column].shift(1)
        return df_copy.reset_index(drop=True)
