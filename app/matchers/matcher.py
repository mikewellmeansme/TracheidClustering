from abc import ABC
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from numpy import ndarray, isnan
from pandas import DataFrame
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Hashable,
)


class Matcher(ABC):

    def boxplot(self, *args, **kwargs) -> Tuple[Optional[Figure], Axes]:
        pass

    def kruskal_wallis_test(self, *args, **kwargs) -> Tuple[float, float]:
        pass

    @staticmethod
    def __get_classes_rows__(df: DataFrame) -> Dict[Hashable, ndarray]:
        groups = df.groupby('Class').groups
        return groups

    @staticmethod
    def __filter_nan__(l: List) -> List:
        result = [el for el in l if not isnan(el)]
        return result
