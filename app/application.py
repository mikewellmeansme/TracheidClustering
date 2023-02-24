import matplotlib.pyplot as plt
import pandas as pd

from dataclasses import dataclass
from itertools import product
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from numpy import (
    array,
    ceil
)
from scipy.stats import mstats
from typing import (
    Optional,
    Dict,
    Tuple,
    List,
    Set
)

from zhutils.dataframes import SuperbDataFrame, DailyDataFrame
from app.matchers import DailyClimateMatcher, MonthlyClimateIndexMatcher, ChronologyMatcher
from app.climate_indexes import AreaIndex, ClimateIndex
from app.normalized_tracheids import NormalizedTracheids
from app.clusterer import Clusterer

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
    normalized_tracheids: NormalizedTracheids
    clusterer: Clusterer
    daily_climate_matcher: DailyClimateMatcher
    monthly_climate_index_matcher: MonthlyClimateIndexMatcher
    chronology_matcher: ChronologyMatcher
    climate_indexes: Dict[str, ClimateIndex]
    clustered_objects: pd.DataFrame
    chronology: pd.DataFrame
    climate_dfs: Dict[str, DailyDataFrame]

    def __init__(
            self,
            tracheid_name: str,
            tracheid_path: str,
            trees: List,
            crn_path: str,
            climate_paths: Dict[str, str],
            climate_index_paths: Optional[Dict[str, str]] = None,
    ) -> None:

        normalized_tracheids = NormalizedTracheids(tracheid_name, tracheid_path, trees)
        self.normalized_tracheids = normalized_tracheids
        self.train_clusterer()

        self.chronology = pd.read_csv(crn_path)
        climate_indexes = {}

        climate_dfs = {}
        for station_name, path in climate_paths.items():
            climate_dfs[station_name] = DailyDataFrame.from_csv(path)
            climate_indexes[f'{station_name}_Area'] = AreaIndex(path)

        self.daily_climate_matcher = DailyClimateMatcher()
        self.monthly_climate_index_matcher = MonthlyClimateIndexMatcher()
        self.chronology_matcher = ChronologyMatcher()

        for index in climate_index_paths:
            climate_indexes[index] = ClimateIndex(index, climate_index_paths[index])

        self.climate_indexes = climate_indexes
        self.climate_dfs = climate_dfs

    def train_clusterer(self, method: str = 'A', nclusters: int = 4) -> None:

        if method.upper() not in 'AB':
            raise Exception(f'Wrong method given! Expected A or B, got {method}!')

        clusterer = Clusterer()
        clusterer.fit(self.normalized_tracheids.obects_for_clustering[f'Method {method}'], nclusters)
        self.clusterer = clusterer

        pred = clusterer.predict(self.normalized_tracheids.obects_for_clustering[f'Method {method}'])

        df = self.normalized_tracheids.obects_for_clustering[f'Method {method}'].copy()
        df['Class'] = pred

        self.clustered_objects = df

    def change_class_names(self, number_to_name: Dict[int, int]) -> None:
        self.clusterer.change_class_names(number_to_name)
        self.clustered_objects['Class'] = self.clusterer.convert_class_number_to_name(self.clustered_objects['Class'])

    def get_class_mean_objects(
            self,
            norm_to=15
    ) -> Dict[int, ClusterMeanObdect]:

        nclasses = self.__get_nclasses__()
        result = dict()

        for i in range(nclasses):
            selected = self.clustered_objects[self.clustered_objects['Class'] == i]
            class_size = len(selected)
            selected_d = selected[[f'D{_ + 1}' for _ in range(norm_to)]]
            selected_cwt = selected[[f'CWT{_ + 1}' for _ in range(norm_to)]]

            d_mean = array(selected_d.mean())
            d_conf_interfal = 1.96 * array(selected_d.std()) / (class_size ** 0.5)

            cwt_mean = array(selected_cwt.mean())
            cwt_conf_interfal = 1.96 * array(selected_cwt.std()) / (class_size ** 0.5)

            result[i] = ClusterMeanObdect(d_mean, d_conf_interfal, cwt_mean, cwt_conf_interfal)

        return result

    def plot_class_mean_objects(
            self,
            class_titles: Optional[List] = None,
            norm_to: int = 15,
            ylim0: float = 0.75,
            ylim1: float = 1.25,
            xticks: List[int] = default_xticks,
            xticklabels: List[int] = default_xticklabels,
            other_mean_objects: Optional[Dict[int, ClusterMeanObdect]] = None,
            other_color: str = 'dimgray'
    ) -> Tuple[Figure, Axes]:

        nclasses = self.__get_nclasses__()

        if class_titles and len(class_titles) != nclasses:
            raise Exception(
                f'Number of class titles ({len(class_titles)}) is not equal to number of classes ({nclasses})!')

        if len(xticks) != len(xticklabels):
            raise Exception(
                f'Length of xticks ({len(xticks)}) is not equal to lenght of xticklabels ({len(xticklabels)})')

        nrows = int(ceil(nclasses / 2))
        ncols = 2

        mean_objects = self.get_class_mean_objects(norm_to)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3 * nrows), dpi=200)
        plt.subplots_adjust(hspace=0.3)

        if nrows > 1:
            axis_positions = product(range(nrows), range(ncols))
        else:
            axis_positions = range(ncols)
        
        d_xrange = range(1, norm_to + 1)
        cwt_xrange = range(norm_to + 2, norm_to * 2 + 2)

        for i, pos in enumerate(axis_positions):
            
            if i >= nclasses:
                break
            
            class_title = f' {class_titles[i]}' if class_titles else ''

            ax = axes[pos]

            ax.axhline(y=1, c='grey', linewidth=1)
            ax.axvline(x=norm_to + 1, c='dimgrey', linewidth=2)

            to_plot = []

            if other_mean_objects:
                to_plot.append(
                    [ax, d_xrange, other_mean_objects[i].d_mean, other_mean_objects[i].d_conf_interfal, other_color])
                to_plot.append([ax, cwt_xrange, other_mean_objects[i].cwt_mean, other_mean_objects[i].cwt_conf_interfal,
                                other_color])

            to_plot.append([ax, d_xrange, mean_objects[i].d_mean, mean_objects[i].d_conf_interfal])
            to_plot.append([ax, cwt_xrange, mean_objects[i].cwt_mean, mean_objects[i].cwt_conf_interfal])

            for args in to_plot:
                self.__plot_mean_obj_with_conf_interfal__(*args)

            ax.set_ylim([ylim0, ylim1])
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_title(f"{i + 1} class{class_title}")
            ax.text(0.25, 0.94, 'D', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.text(0.75, 0.94, 'CWT', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.text(0.1, 0.95, f'{chr(65 + i)})', transform=ax.transAxes, fontsize=16, va='top', ha='right')

        if nrows > 1:
            for i in range(nrows):
                axes[i, 0].set_ylabel("Deviation (rel. units)")
            axes[nrows - 1, 0].set_xlabel("Standardized cell position")
            axes[nrows - 1, 1].set_xlabel("Standardized cell position")
        else:
            axes[0].set_ylabel("Deviation (rel. units)")
            axes[0].set_xlabel("Standardized cell position")
            axes[1].set_xlabel("Standardized cell position")
        return fig, axes

    @staticmethod
    def __plot_mean_obj_with_conf_interfal__(
            ax: Axes,
            xrange: List,
            mean_obj: array,
            conf_interfal: array,
            color: str = 'k'
    ) -> None:
        ax.plot(xrange, mean_obj + conf_interfal, c=color, linestyle='--', linewidth=1)
        ax.plot(xrange, mean_obj - conf_interfal, c=color, linestyle='--', linewidth=1)
        ax.plot(xrange, mean_obj, c=color)

    def get_class_sizes(self) -> Dict[int, int]:
        classes = set(self.clustered_objects['Class'])
        result = dict()

        for cl in classes:
            selected = self.clustered_objects[self.clustered_objects['Class'] == cl]
            result[cl] = len(selected)

        return result

    def plot_area_per_class(self, station_name: str, **kwargs) -> Tuple[Figure, Axes]:

        fig, ax = self.climate_indexes[f'{station_name}_Area'].plot_area_per_class(
            clustered_objects=self.clustered_objects,
            **kwargs
        )

        return fig, ax

    def plot_class_boxplot(self, feature: str = 'D') -> Tuple[Figure, Axes]:
        r"""
        Params:
            feature: D or CWT
        """
        nclasses = self.__get_nclasses__()
        groups = self.__get_classes_rows__()
        columns = self.__get_feature_columns__(feature)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=200)
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

    def boxplot_climate_index(self, climate_index: str, **kwargs) -> Tuple[Figure, Axes]:
        fig, ax = self.monthly_climate_index_matcher.boxplot(
            climate_index=self.climate_indexes[climate_index],
            classes_df=self.clustered_objects,
            **kwargs)
        return fig, ax

    def get_climate_index_kruskalwallis(self, climate_index: str, **kwargs) -> Tuple[float, float]:
        s, p = self.monthly_climate_index_matcher.kruskal_wallis_test(
            climate_index=self.climate_indexes[climate_index],
            classes_df=self.clustered_objects,
            **kwargs
        )
        return s, p

    def get_chronology_comparison(self, months, crn_column) -> SuperbDataFrame:
        result = self.chronology_matcher.get_chronology_comparison(
            self.chronology,
            self.climate_indexes,
            months,
            crn_column,
            self.clustered_objects
        )
        return result

    def plot_chronology_comparison(self, months, crn_column, **kwargs) -> Tuple[Figure, Axes]:
        crn_comparison_df = self.get_chronology_comparison(months, crn_column)
        fig, ax = self.chronology_matcher.plot_chronology_comparison(
            crn_comparison_df,
            self.climate_indexes,
            crn_column,
            **kwargs
        )
        return fig, ax

    def boxplot_climate(self, station_name: str, **kwargs) -> Tuple[Figure, Axes]:
        fig, ax = self.daily_climate_matcher.boxplot(
            self.climate_dfs[station_name],
            self.clustered_objects,
            **kwargs
        )
        return fig, ax

    def get_climate_kruskalwallis(self, station_name: str, **kwargs) -> Tuple[float, float]:
        s, p = self.daily_climate_matcher.kruskal_wallis_test(
            self.climate_dfs[station_name],
            self.clustered_objects,
            **kwargs
        )

        return s, p

    def get_climate_comparison(self, station_name: str, **kwargs) -> SuperbDataFrame:
        result = self.daily_climate_matcher.get_climate_comparison(
            self.climate_dfs[station_name],
            self.clustered_objects,
            **kwargs
        )

        return result

    def __get_nclasses__(self) -> int:
        classes = set(self.clustered_objects['Class'])
        nclasses = len(classes)
        return nclasses

    def __get_classes_rows__(self) -> Dict[int, list]:
        groups = self.clustered_objects.groupby('Class').groups
        return groups

    def __get_feature_columns__(self, feature) -> List[str]:

        if feature not in ['D', 'CWT']:
            raise ValueError(f'Wrong feature given! Must be D or CWT given: {feature}')

        columns = [column for column in self.clustered_objects.columns if feature in column]
        return columns
