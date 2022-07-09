from matcher import Matcher
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from typing import Optional, Tuple 


class DailyClimateMatcher(Matcher):
    
    def boxplot(self, *args, **kwargs) -> Tuple[Optional[Figure], Axes]:
        pass

    def kruskal_wallis_test(self, *args, **kwargs) -> Tuple[float, float]:
        pass
