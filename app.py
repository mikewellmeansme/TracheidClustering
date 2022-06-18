import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from datetime import date
from itertools import product
from numpy import (
    array,
    ceil
)


from normalized_tracheids import NormalizedTracheids
from climate_matcher import ClimateMatcher
from clusterer import Clusterer
from zhutils.daily_dataframe import DailyDataFrame


pd.options.mode.chained_assignment = None

default_xticks = [1, 5, 10, 15, 17, 21, 26, 31]
default_xticklabels = [1, 5, 10, 15, 1, 5, 10, 15]


class Application:
    normalized_tracheids : NormalizedTracheids
    clusterer : Clusterer
    climate_matcher: ClimateMatcher
    clustered_objects: pd.DataFrame


    def __init__(self, tracheid_name: str,  tracheid_path: str,
                 trees: list, climate_path: str) -> None:

        normalized_tracheids = NormalizedTracheids(tracheid_name, tracheid_path, trees)
        self.normalized_tracheids = normalized_tracheids
        self.train_clusterer()

        self.climate_matcher = ClimateMatcher(climate_path, self.clustered_objects)
    

    def train_clusterer(self, method: str ='A', nclusters: int = 4) -> None:

        if method.upper() not in 'AB':
            raise Exception(f'Wrong method given! Expected A or B, got {method}!')

        clusterer = Clusterer()
        clusterer.fit(self.normalized_tracheids.obects_for_clustering[f'Method {method}'], nclusters)
        self.clusterer = clusterer

        pred = clusterer.predict(self.normalized_tracheids.obects_for_clustering[f'Method {method}'])

        df = self.normalized_tracheids.obects_for_clustering[f'Method {method}'].copy()
        df['Class'] = pred

        self.clustered_objects = df

    
    def change_class_names(self, number_to_name: dict[int, int]) -> None:
        self.clusterer.change_class_names(number_to_name)
        self.clustered_objects['Class'] = self.clusterer.convert_class_number_to_name(self.clustered_objects['Class'])
        self.climate_matcher.change_class_names(clusterer=self.clusterer)


    def plot_сlass_mean_objects(self, class_titles: list = None,
                                norm_to=15, ylim0=.75, ylim1=1.25,
                                xticks=default_xticks, xticklabels=default_xticklabels) -> tuple:

        classes = set(self.clustered_objects['Class'])
        nclasses = len(classes)

        if class_titles and len(class_titles) != nclasses:
            raise Exception(f'Number of class titles ({len(class_titles)}) is not equal to number of classes ({nclasses})!')
        
        if len(xticks) != len(xticklabels):
            raise Exception(f'Length of xticks ({len(xticks)}) is not equal to lenght of xticklabels ({len(xticklabels)})')
        
        nrows = int(ceil(nclasses/2))
        ncols = 2

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols,3*nrows), dpi=200)
        plt.subplots_adjust(hspace=0.3)

        axis_positions = product(range(nrows), range(ncols))

        for i, pos in enumerate(axis_positions):
            selected = self.clustered_objects[self.clustered_objects['Class']==i]
            class_size = len(selected)
            selected_d = selected[[f'D{_+1}' for _ in range(norm_to)]]
            selected_cwt = selected[[f'CWT{_+1}' for _ in range(norm_to)]]
            class_title = f' {class_titles[i]}' if class_titles else ''

            ax = axes[pos]

            ax.axhline(y=1, c='grey', linewidth=1)
            ax.axvline(x=norm_to+1, c='dimgrey', linewidth=2 )
            
            d_xrange = range(1, norm_to+1)
            d_mean = array(selected_d.mean())
            d_conf_interfal = 1.96 * array(selected_d.std()) / (class_size ** 0.5)
            
            ax.plot(d_xrange, d_mean + d_conf_interfal, c='k', linestyle='--', linewidth=1)
            ax.plot(d_xrange, d_mean - d_conf_interfal, c='k', linestyle='--', linewidth=1)
            ax.plot(d_xrange, d_mean, c='k')

            cwt_xrange = range(norm_to+2, norm_to*2+2)
            cwt_mean = array(selected_cwt.mean())
            cwt_conf_interfal = 1.96 * array(selected_cwt.std()) / (class_size ** 0.5)
            
            ax.plot(cwt_xrange, cwt_mean + cwt_conf_interfal, c='k', linestyle='--', linewidth=1)
            ax.plot(cwt_xrange, cwt_mean - cwt_conf_interfal, c='k', linestyle='--', linewidth=1)
            ax.plot(cwt_xrange, cwt_mean, c='k')

            ax.set_ylim([ylim0, ylim1])
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_title(f"{i+1} class{class_title}")
            ax.text(0.25, 0.94, 'D', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.text(0.75, 0.94, 'CWT', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.text(0.1, 0.95, f'{chr(65+i)})', transform=ax.transAxes, fontsize=16, va='top', ha='right')

        for i in range(nrows):
            axes[i, 0].set_ylabel("Deviation (rel. units)")
        axes[nrows-1, 0].set_xlabel("Standardized cell position")
        axes[nrows-1, 1].set_xlabel("Standardized cell position")

        return fig, axes


    def get_class_sizes(self) -> dict[int, int]:
        classes = set(self.clustered_objects['Class'])
        result = dict()

        for cl in classes:
            selected = self.clustered_objects[self.clustered_objects['Class']==cl]
            result[cl] = len(selected)
        
        return result
    
    
    def plot_area_per_class(self, xlim: list = [date(2000, 4, 20), date(2000, 10, 10)],
                            temp_ylim: list = [0, 30], prec_ylim: list = [0,350]):
        
        fig, ax = self.climate_matcher.plot_area_per_class(
            clustered_objects=self.clustered_objects,
            xlim=xlim,
            temp_ylim=temp_ylim,
            prec_ylim=prec_ylim
        )

        return fig, ax
