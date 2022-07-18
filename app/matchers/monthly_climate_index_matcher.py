import matplotlib.pyplot as plt
import pandas as pd

from matchers.matcher import Matcher
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from scipy.stats import mstats
from typing import (
    Dict,
    Optional,
    Tuple,
    List
)

from climate_indexes.climate_index import ClimateIndex


class MonthlyClimateIndexMatcher(Matcher):

    def get_values_per_class(
            self,
            climate_index: ClimateIndex,
            classes_df: pd.DataFrame,
            prev: bool = False,
            month: Optional[str] = None,
            classes: Optional[List] = None
    ) -> Dict[int, List]:

        r"""
        Получает все значения индексов по классам
        Params:
            classes_df: Dataframe with 'Class' and 'Year' columns.
            prev: Флаг того, сравнивается ли климатика этого года или предыдущего
            month: Месяц, по которому сравниваем. По-умолчанию сравниваем по колонке с названием индекса.
            classes: Классы, для которых происходит сравнение. По-умолчанию: все.
        
        Возвращает словарь, где в ключах номер класса, а в значениях, все значения климатического индекса для этого класса
        """

        classes = classes if classes else set(classes_df['Class'])

        df = self.__merge_with_classes__(climate_index.climate_index, classes_df)
        groups = self.__get_classes_rows__(df)

        if prev:
            df = self.__get_shifted_df__(df)

        column = month if month else climate_index.name
        result = {j: self.__filter_nan__(list(df.loc[groups[j]][column])) for j in classes}

        return result

    def boxplot(
            self,
            climate_index: ClimateIndex,
            classes_df: pd.DataFrame,
            prev: bool = False,
            month: Optional[str] = None,
            classes: Optional[List] = None,
            ylims: Optional[List] = None
    ) -> Tuple[Figure, Axes]:

        r"""
        Params:
        Строит boxplot индекса по классам
            classes_df: Dataframe with 'Class' and 'Year' columns.
            prev: Флаг того, сравнивается ли климатика этого года или предыдущего
            month: Месяц, по которому сравниваем. По-умолчанию сравниваем по колонке с названием индекса.
            classes: Классы, для которых происходит сравнение. По-умолчанию: все.
            ylims: Пределы по оси Y.
        """

        classes = classes if classes else set(classes_df['Class'])

        values_per_class = self.get_values_per_class(
            climate_index,
            classes_df,
            prev,
            month,
            classes
        )

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=200)
        ax.boxplot(values_per_class.values())
        title = f"{climate_index.name} {month if month else ''}{' prev' if prev else ''}"
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
            month: Optional[str] = None,
            classes: Optional[List] = None
    ) -> Tuple[float, float]:

        r"""
        Params:
            classes_df: Dataframe with 'Class' and 'Year' columns
            prev: Флаг того, сравнивается ли климатика этого года или предыдущего
            month: Месяц, по которому сравниваем. По-умолчанию сравниваем по колонке с названием индекса.
            classes: Классы, для которых происходит сравнение. По-умолчанию: все.
        """

        classes = classes if classes else set(classes_df['Class'])

        values_per_class = self.get_values_per_class(
            climate_index,
            classes_df,
            prev,
            month,
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
