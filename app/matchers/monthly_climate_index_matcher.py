from matcher import Matcher
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from typing import Optional, Tuple 


class MonthlyClimateIndexMatcher(Matcher):

    def boxplot(self, *args, **kwargs) -> Tuple[Optional[Figure], Axes]:
        pass

    def kruskal_wallis_test(self, *args, **kwargs) -> Tuple[float, float]:
        pass