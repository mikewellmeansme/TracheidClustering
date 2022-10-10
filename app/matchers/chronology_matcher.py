import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from numpy import arange
from typing import (
    Dict, 
    Tuple,
    List
) 
from zhutils.dataframes import SuperbDataFrame

from climate_indexes import ClimateIndex
from matchers.matcher import Matcher


class ChronologyMatcher(Matcher):

    @staticmethod
    def get_chronology_comparison(
            chronology: pd.DataFrame,
            climate_indexes: Dict[str, ClimateIndex],
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
            [climate_indexes[index].climate_index[['Year', index]].set_index('Year') for index in climate_indexes],
            how='left'
        ).reset_index()
        result = clustered_objects[['Year', 'Class']].merge(df, on='Year', how='left')
        return SuperbDataFrame(result)
    
    @staticmethod
    def plot_chronology_comparison(
            crn_comparison_df: pd.DataFrame,
            climate_indexes: Dict[str, ClimateIndex],
            crn_column: str,
            ylims: Dict[str, List] = {
                'std': [0, 2],
                'PDSI': [-9, 9],
                'SPEI': [-1.5, 1.5],
                'Area': [20, 80],
                'Class': [0.5,4.5]
            },
            yticks: Dict[str, List] = {
                'std': arange(0.5,2, 0.5),
                'Area': [25, 50, 75],
                'Class': range(1,5)
            }
        ) -> Tuple[Figure, Axes]:

        r"""
        Params:
            crn_comparison_df: 
            crn_column: 
            ylims:
            yticks:
        """

        nrows = len(climate_indexes) + 2
        fig, axes = plt.subplots( nrows=nrows, ncols=1, dpi=300, figsize=(10,nrows), sharex=True)
        plt.subplots_adjust(hspace=0, left=.06, right=.98, top=.99, bottom=.09)

        row_names = [crn_column] + list(climate_indexes.keys()) + ['Class']
        

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
