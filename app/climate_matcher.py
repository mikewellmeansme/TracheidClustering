import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from datetime import date
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from numpy import arange
from scipy.stats import mstats
from typing import (
    Optional,
    Dict, 
    Tuple,
    List,
    Set
) 
from zhutils.daily_dataframe import DailyDataFrame
from zhutils.superb_dataframe import SuperbDataFrame

from clusterer import Clusterer


class ClimateMatcher:
    climate: DailyDataFrame
    growth_season_start_month: int = 5
    growth_season_end_month: int = 9
    cut_сlimate: pd.DataFrame
    area: pd.DataFrame
    clustered_objects: pd.DataFrame
    climate_indexes: dict[pd.DataFrame]


    def __init__(
            self,
            climate_path: str,
            clustered_objects: pd.DataFrame, 
            climate_indexes_paths: Dict[str, str]
        ) -> None:
        
        climate = pd.read_csv(climate_path)
        self.climate = DailyDataFrame(climate)
        self.__get_area_index__(clustered_objects)
        self.climate_indexes = {
            'Area': self.area
        }
        for index in climate_indexes_paths:
            self.climate_indexes[index] = self.__load_climate_index__(
                climate_indexes_paths[index],
                clustered_objects
            ) 
        

    def __load_climate_index__(self, path: str, clustered_objects: pd.DataFrame):
        df = pd.read_csv(path)
        result = df.merge(clustered_objects[['Year', 'Class']], on='Year', how='left')
        return result
    

    def change_class_names(self, clusterer: Clusterer) -> None:
        for index in self.climate_indexes:
            self.climate_indexes[index]['Class'] = clusterer.convert_class_number_to_name(
                self.climate_indexes[index]['Class']
            )
    

    def get_chronology_comparison(
            self,
            chronology: pd.DataFrame,
            crn_column: str,
            clustered_objects: pd.DataFrame
        ) -> SuperbDataFrame:

        r"""
        Params:
            chronology: 
            crn_column: 
            clustered_objects:
        """

        df = chronology[['Year', crn_column]].set_index('Year').join(
            [self.climate_indexes[index][['Year', index]].set_index('Year') for index in self.climate_indexes],
            how='left'
        ).reset_index()
        result = clustered_objects[['Year', 'Class']].merge(df, on='Year', how='left')
        return SuperbDataFrame(result)
    

    def plot_chronology_comparison(
            self,
            crn_comparison_df: pd.DataFrame,
            crn_column: str,
            ylims: Dict = {
                'std': [0, 2],
                'PDSI': [-9, 9],
                'SPEI': [-1.5, 1.5],
                'Area': [20, 80],
                'Class': [0.5,4.5]
            },
            yticks: Dict = {
                'std': arange(0.5,2, 0.5),
                'Area': [25, 50, 75],
                'Class': range(1,5)
            }
        ) -> tuple:

        r"""
        Params:
            crn_comparison_df: 
            crn_column: 
            ylims:
            yticks:
        """

        nrows = len(self.climate_indexes) + 2
        fig, axes = plt.subplots( nrows=nrows, ncols=1, dpi=300, figsize=(10,nrows), sharex=True)
        plt.subplots_adjust(hspace=0, left=.06, right=.98, top=.99, bottom=.09)

        row_names = [crn_column] + list(self.climate_indexes.keys()) + ['Class']
        

        for i, row in enumerate(row_names):
            y = crn_comparison_df[row] + 1 if row == 'Class' else crn_comparison_df[row]
            axes[i].plot(crn_comparison_df['Year'], y, c='black', label=row)
            axes[i].set_ylabel(row)

            ylim = ylims.get(row)

            if not isinstance(ylim, type(None)):
                axes[i].set_ylim(ylim)
            
            ytick = yticks.get(row)

            if not isinstance(ytick, type(None)):
                axes[i].set_yticks(ytick)

        return fig, axes

    
    def __get_climate_index_names__(self):
        result = list(self.climate_indexes.keys())
        return result
    

    def __validate_inedx__(self, index):
        if index not in self.climate_indexes:
            indexes = ', '.join(self.__get_climate_index_names__())
            raise ValueError(f'Wrong feature given! Must be one of: {indexes}. Given: {index}')
    

    def __get_classes__(self) -> Set[int]:
        classes = set(self.area['Class'])
        return classes


    def __get_nclasses__(self) -> int:
        classes = self.__get_classes__()
        nclasses = len(classes)
        return nclasses
