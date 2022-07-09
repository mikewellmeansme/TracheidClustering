from abc import ABC
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from pandas import DataFrame
from typing import (
    Dict,
    Optional,
    Tuple
)


class Matcher(ABC):

    def boxplot(self, *args, **kwargs) -> Tuple[Optional[Figure], Axes]:
        pass

    def kruskal_wallis_test(self, *args, **kwargs) -> Tuple[float, float]:
        pass

    @staticmethod
    def __get_classes_rows__(df: DataFrame) -> Dict[int, list]:
        groups = df.groupby('Class').groups
        return groups
