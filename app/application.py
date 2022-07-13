import matplotlib.pyplot as plt
import pandas as pd

from dataclasses import dataclass
from itertools import product
from numpy import (
    array,
    ceil
)
from scipy.stats import mstats
from zhutils.superb_dataframe import SuperbDataFrame

from normalized_tracheids import NormalizedTracheids
from climate_matcher import ClimateMatcher
from clusterer import Clusterer


pd.options.mode.chained_assignment = None

default_xticks = [1, 5, 10, 15, 17, 21, 26, 31]
default_xticklabels = [1, 5, 10, 15, 1, 5, 10, 15]


@dataclass
class ClusterMeanObdect:
    d_mean: array
    d_conf_interfal: array
    cwt_mean: array
    cwt_conf_interfal: array


class Application:
    normalized_tracheids : NormalizedTracheids
    clusterer : Clusterer
    climate_matcher: ClimateMatcher
    clustered_objects: pd.DataFrame
    chronology: pd.DataFrame


    def __init__(
            self,
            tracheid_name: str,
            tracheid_path: str,
            trees: list,
            climate_path: str,
            climate_indexes_paths: dict[str, str],
            crn_path: str
        ) -> None:

        normalized_tracheids = NormalizedTracheids(tracheid_name, tracheid_path, trees)
        self.normalized_tracheids = normalized_tracheids
        self.train_clusterer()

        self.chronology = pd.read_csv(crn_path)

        self.climate_matcher = ClimateMatcher(climate_path, self.clustered_objects, climate_indexes_paths)
    

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
    

    def get_class_mean_objects(
            self,
            norm_to=15
        ) -> dict[ClusterMeanObdect]:
        
        nclasses = self.__get_nclasses__()
        result = dict()

        for i in range(nclasses):
            selected = self.clustered_objects[self.clustered_objects['Class']==i]
            class_size = len(selected)
            selected_d = selected[[f'D{_+1}' for _ in range(norm_to)]]
            selected_cwt = selected[[f'CWT{_+1}' for _ in range(norm_to)]]
            
            d_mean = array(selected_d.mean())
            d_conf_interfal = 1.96 * array(selected_d.std()) / (class_size ** 0.5)

            cwt_mean = array(selected_cwt.mean())
            cwt_conf_interfal = 1.96 * array(selected_cwt.std()) / (class_size ** 0.5)

            result[i] = ClusterMeanObdect(d_mean, d_conf_interfal, cwt_mean, cwt_conf_interfal)
        
        return result


    def plot_сlass_mean_objects(
            self,
            class_titles: list = None,
            norm_to=15,
            ylim0=.75,
            ylim1=1.25,
            xticks=default_xticks,
            xticklabels=default_xticklabels
        ) -> tuple:

        nclasses = self.__get_nclasses__()

        if class_titles and len(class_titles) != nclasses:
            raise Exception(f'Number of class titles ({len(class_titles)}) is not equal to number of classes ({nclasses})!')
        
        if len(xticks) != len(xticklabels):
            raise Exception(f'Length of xticks ({len(xticks)}) is not equal to lenght of xticklabels ({len(xticklabels)})')
        
        nrows = int(ceil(nclasses/2))
        ncols = 2

        mean_objects = self.get_class_mean_objects(norm_to)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols,3*nrows), dpi=200)
        plt.subplots_adjust(hspace=0.3)

        axis_positions = product(range(nrows), range(ncols))
        d_xrange = range(1, norm_to+1)
        cwt_xrange = range(norm_to+2, norm_to*2+2)

        for i, pos in enumerate(axis_positions):
            class_title = f' {class_titles[i]}' if class_titles else ''

            ax = axes[pos]

            ax.axhline(y=1, c='grey', linewidth=1)
            ax.axvline(x=norm_to+1, c='dimgrey', linewidth=2 )
            
            d_mean = mean_objects[i].d_mean
            d_conf_interfal = mean_objects[i].d_conf_interfal
            
            ax.plot(d_xrange, d_mean + d_conf_interfal, c='k', linestyle='--', linewidth=1)
            ax.plot(d_xrange, d_mean - d_conf_interfal, c='k', linestyle='--', linewidth=1)
            ax.plot(d_xrange, d_mean, c='k')

            cwt_mean = mean_objects[i].cwt_mean
            cwt_conf_interfal = mean_objects[i].cwt_conf_interfal
            
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
    

    def plot_area_per_class(self, **kwargs) -> tuple:
        
        fig, ax = self.climate_matcher.plot_area_per_class(
            clustered_objects=self.clustered_objects,
            **kwargs
        )

        return fig, ax
    

    def plot_class_boxplot(self, feature: str = 'D') -> tuple:
        r"""
        Params:
            feature: D or CWT
        """
        nclasses = self.__get_nclasses__()
        groups = self.__get_classes_rows__()
        columns = self.__get_feature_columns__(feature)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), dpi=200)
        ax.boxplot([list(self.clustered_objects.loc[groups[j], columns].mean(axis=1)) for j in range(nclasses)])
        ax.set_title(feature)
        
        return fig, ax
    

    def get_class_kruskalwallis(self, feature: str = 'D') -> pd.DataFrame:
        r"""
        Params:
            feature: D or CWT
        """
        nclasses = self.__get_nclasses__()
        groups = self.__get_classes_rows__()
        columns = self.__get_feature_columns__(feature)

        stats, p_values = [], []

        for column in columns:
            s, p = mstats.kruskalwallis(
                *[list(self.clustered_objects.loc[groups[j], column]) for j in range(nclasses)]
            )
            stats.append(s)
            p_values.append(p)
        
        result = pd.DataFrame({
            'Feature': columns,
            'Statistic': stats,
            'P-values': p_values
        })
        return result
    

    def boxplot_climate_index(self, *args, **kwargs) -> tuple:
        fig, ax = self.climate_matcher.boxplot_climate_index(*args, **kwargs)
        return fig, ax
    

    def get_climate_index_kruskalwallis(self, *args, **kwargs) -> tuple[float, float]:
        s, p = self.climate_matcher.get_climate_index_kruskalwallis(*args, **kwargs)
        return s, p
    

    def get_chronology_comparison(self, crn_column) -> SuperbDataFrame:
        result = self.climate_matcher.get_chronology_comparison(
            self.chronology,
            crn_column,
            self.clustered_objects
        )
        return result
    

    def plot_chronology_comparison(self, crn_column, **kwargs) -> tuple:
        crn_comparison_df = self.get_chronology_comparison(crn_column)
        fig, ax = self.climate_matcher.plot_chronology_comparison(
            crn_comparison_df,
            crn_column,
            **kwargs
        )
        return fig, ax
    

    def boxplot_climate(self, **kwargs) -> tuple:
        fig, ax = self.climate_matcher.boxplot_climate(
            self.clustered_objects, **kwargs
        )
        return fig, ax
    

    def get_climate_kruskalwallis(self, **kwargs) -> tuple[float, float]:
        s, p = self.climate_matcher.get_climate_kruskalwallis(
            self.clustered_objects, **kwargs
        )

        return s, p
    

    def get_climate_comparison(self, **kwargs) -> SuperbDataFrame:
        result = self.climate_matcher.get_climate_comparison(
            self.clustered_objects,
            **kwargs
        )

        return result

    
    def __get_nclasses__(self) -> int:
        classes = set(self.clustered_objects['Class'])
        nclasses = len(classes)
        return nclasses
    

    def __get_classes_rows__(self) -> dict[int, list]:
        groups = self.clustered_objects.groupby('Class').groups
        return groups
    
    
    def __get_feature_columns__(self, feature) -> list[str]:

        if feature not in ['D', 'CWT']:
            raise ValueError(f'Wrong feature given! Must be D or CWT given: {feature}')
        
        columns = [column for column in self.clustered_objects.columns if feature in column]
        return columns
