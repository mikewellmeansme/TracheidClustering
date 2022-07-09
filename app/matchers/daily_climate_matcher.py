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
from zhutils.daily_dataframe import DailyDataFrame
from zhutils.superb_dataframe import SuperbDataFrame


class DailyClimateMatcher(Matcher):
    climate: DailyDataFrame

    def __init__(self, climate_path: str) -> None:
        climate = pd.read_csv(climate_path)
        self.climate = DailyDataFrame(climate)


    def boxplot(
            self,
            clustered_objects: pd.DataFrame,
            ylim: Optional[List] = None,
            column: str = 'Temperature',
            classes: Optional[List] = None,
            start_month = 5,
            end_month = 9,
            start_day = 1,
            end_day = 31,
            moving_avg_window: Optional[int] = None,
            prev: bool = False,
            ax: Optional[Axes] = None
        ) -> Tuple[Optional[Figure], Axes]:

        r"""
        Plots a boxplot of mean temperature \ total precipitation
        for the given period (from start_month.start_day till end_month.end_day)
        for every year
        per classes (given in clustered_objects)

        Params:
            clustered_objects: DataFrame with columns 'Year' and 'Class'
            ylim: Limits for the Y axis of boxplot
            column: Temperature or Precipitation
            classes: List of classes to plot (default: all)
            start_month: Start month of period
            end_month: End month of period
            start_day: Start day of period
            end_day: End day of period
            moving_avg_window: Window of the moving average to smooth the climate
            prev: True if we are dealing with the previous years (defaulf: False)
            ax: Axis to plot on (default: None, method creates a new axis)
        """

        classes = classes if classes else set(clustered_objects['Class'])

        totals = self.__get_total_climate__(
            clustered_objects,
            column,
            classes,
            start_month,
            end_month,
            start_day,
            end_day,
            moving_avg_window,
            prev
        )
        if ax:
            fig = None
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), dpi=200)
            ylabel = 'Mean Temperature (°C)' if column == 'Temperature' else 'Total Precipitation (mm)'
            ax.set_xticklabels([cl + 1 for cl in classes])
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Class')
        
        ax.boxplot(totals)

        if ylim:
            ax.set_ylim(ylim)

        return fig, ax
    

    def kruskal_wallis_test(
            self,
            clustered_objects: pd.DataFrame,
            column: str = 'Temperature',
            classes: Optional[List] = None,
            start_month = 5,
            end_month = 9,
            start_day = 1,
            end_day = 31,
            moving_avg_window: Optional[int] = None,
            prev: bool = False
        ) -> Tuple[float, float]:
        
        r"""
        Calculates the Kruskall-Wallis stat for partition of mean temperature \ total precipitation
        for the given period (from start_month.start_day till end_month.end_day)
        for every year 
        per classes (given in clustered_objects)

        Params:
            clustered_objects: DataFrame with columns 'Year' and 'Class'
            column: Temperature or Precipitation
            classes: List of classes to plot (default: all)
            start_month: Start month of period
            end_month: End month of period
            start_day: Start day of period
            end_day: End day of period
            moving_avg_window: Window of the moving average to smooth the climate
            prev: True if we are dealing with the previous years (defaulf: False)
        """

        totals = self.__get_total_climate__(
            clustered_objects,
            column,
            classes,
            start_month,
            end_month,
            start_day,
            end_day,
            moving_avg_window,
            prev
        )

        s, p = mstats.kruskalwallis(*totals)

        return s, p
    

    def get_climate_comparison(
            self,
            clustered_objects: pd.DataFrame,
            start_month = 5,
            end_month = 9
        ) -> SuperbDataFrame:

        r"""
        Params:
            clustered_objects:
            start_month:
            end_month:
        """

        crn = self.climate[
            (start_month <= self.climate['Month']) & 
            (self.climate['Month'] <= end_month)
        ].groupby('Year').mean().reset_index()
        
        result = crn.merge(
            clustered_objects[['Year', 'Class']],
            on='Year', 
            how='left'
        ).drop(columns=['Month', 'Day'])

        return SuperbDataFrame(result)
    

    def __get_total_climate__(
            self,
            clustered_objects: pd.DataFrame,
            column: str,
            classes: List,
            start_month: int,
            end_month: int,
            start_day: int,
            end_day: int,
            moving_avg_window: Optional[int] = None,
            prev: bool = False

        ) -> List[List[float]]:

        r"""
        Returns the partition of mean temperature \ total precipitation
        for the given period (from start_month.start_day till end_month.end_day)
        for every year 
        per classes (given in clustered_objects)

        Params:
            clustered_objects: DataFrame with columns 'Year' and 'Class'
            column: Temperature or Precipitation
            classes: List of classes to plot (default: all)
            start_month: Start month of period
            end_month: End month of period
            start_day: Start day of period
            end_day: End day of period
            moving_avg_window: Window of the moving average to smooth the climate
            prev: True if we are dealing with the previous years (defaulf: False)
        """

        totals = []

        classes = classes if classes else set(clustered_objects['Class'])
        groups = self.__get_classes_rows__(clustered_objects)

        climate_df = DailyDataFrame(self.climate.copy())
        if moving_avg_window:
            moving_avg = climate_df.moving_avg([column], window=moving_avg_window)
            climate_df[column] = moving_avg[column]

        for c in classes:
            values = []
            years = clustered_objects.loc[groups[c]]['Year']
            for year in years:
                df = climate_df[
                    (climate_df['Year'] == year - int(prev)) &
                    (
                        (
                            ((start_month == climate_df['Month']) & (start_day <= climate_df['Day'])) |
                            (start_month < climate_df['Month'])
                        ) & 
                        (
                            (climate_df['Month'] < end_month) |
                            ((climate_df['Month'] ==  end_month) & (climate_df['Day'] <= end_day))
                        )
                    )
                ].dropna()[column]

                if len(df) > 0:
                    res = df.mean() if column == 'Temperature' else df.sum()
                    values.append(res)
            totals.append(values)
        
        return totals
