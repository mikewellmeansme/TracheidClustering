import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from itertools import product
from numpy import (
    array,
    ceil
)


from normalized_tracheids import NormalizedTracheids
from clusterer import Clusterer
from zhutils.daily_dataframe import DailyDataFrame
from zhutils.superb_dataframe import SuperbDataFrame


pd.options.mode.chained_assignment = None

default_xticks = [1, 5, 10, 15, 17, 21, 26, 31]
default_xticklabels = [1, 5, 10, 15, 1, 5, 10, 15]


class Application:
    normalized_tracheids : NormalizedTracheids
    clusterer : Clusterer
    clustered_objects: pd.DataFrame
    climate: DailyDataFrame
    growth_season_start_month: int = 5
    growth_season_end_month: int = 9
    cut_сlimate: pd.DataFrame
    area: pd.DataFrame


    def __init__(self, tracheid_name,  tracheid_path, trees,
                 climate_path) -> None:
        climate = pd.read_csv(climate_path)
        self.climate = DailyDataFrame(climate)

        normalized_tracheids = NormalizedTracheids(tracheid_name, tracheid_path, trees)
        self.normalized_tracheids = normalized_tracheids
        self.train_clusterer()
        self.__get_area_index__()
    

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
    

    def __get_cut_climate__(self) -> None:
        r"""
        Function for creating a climate dataframe which was cut from the start to the end of growth season
        and contains:
            rolled temperature
            cumulative presipitation
            scaled rolled temperature
            scaled cumulative presipitation
            absolute value of difference between scaled temperature and presipitation
        """
        climate_df = DailyDataFrame(self.climate.copy())
        # Turning daily precipitation to cumulative presipitetion
        climate_df['Prec_cumsum'] = climate_df.fillna(0).groupby('Year')['Precipitation'].cumsum()

        # Smoothing temperature and cumulative precipitation with 7-day mooving average
        moving_avg = climate_df.moving_avg(['Temperature', 'Prec_cumsum'])
        climate_df['Temp_rolling'] = moving_avg['Temperature']
        climate_df['Prec_cumsum_rolling'] = moving_avg['Prec_cumsum']

        # Cutting climate from the start to the end of growth season
        start = self.growth_season_start_month
        end = self.growth_season_end_month
        cut_сlimate_df = climate_df[(start <= climate_df['Month']) & (climate_df['Month'] <= end)]
        cut_сlimate_df = cut_сlimate_df.reset_index(drop=True)

        temp_max = cut_сlimate_df['Temp_rolling'].max()
        temp_min = cut_сlimate_df['Temp_rolling'].min()
        prec_max = cut_сlimate_df['Prec_cumsum_rolling'].max()
        prec_min = cut_сlimate_df['Prec_cumsum_rolling'].min()

        # Scaling temperature and precipitation with MinMax approach
        cut_сlimate_df['Temp_scaled'] = (cut_сlimate_df['Temp_rolling'] - temp_min) / (temp_max - temp_min)
        cut_сlimate_df['Prec_scaled'] = (cut_сlimate_df['Prec_cumsum_rolling'] - prec_min) / (prec_max - prec_min)
        
        # Calculating the difference between scaled temperature and presipitation
        cut_сlimate_df['Temp_prec_difference'] = abs(cut_сlimate_df['Temp_scaled'] - cut_сlimate_df['Prec_scaled'])

        self.cut_сlimate = cut_сlimate_df


    def __get_area_index__(self) -> None:
        r"""
        Function for calculating Area Index for the given climate data
        """
        self.__get_cut_climate__()
        cut_сlimate_df = self.cut_сlimate
        area_df = cut_сlimate_df[['Year', 'Temp_prec_difference']].groupby('Year').sum().reset_index()
        area_df = area_df.rename(columns={'Temp_prec_difference': 'Area'})

        area_df = area_df.merge(self.clustered_objects[['Year', 'Class']], on='Year', how='left')
        self.area = area_df
    

    def plot_area_per_class(self):
        classes = set(self.clustered_objects['Class'])
        nclasses = len(classes)

        fig, ax = plt.subplots(nrows=nclasses, ncols=2, dpi=300, figsize=(10, 3 * nclasses))
        plt.subplots_adjust(bottom=0.03, top=0.95)

        for i in range(nclasses):
            selected = SuperbDataFrame(self.area[self.area['Class'] == i])
            median_year_index = selected.median_index()['Area']
            median_year = int(selected.loc[median_year_index]['Year'])

            median_year_climate = self.cut_сlimate[self.cut_сlimate['Year'] == median_year]

            date_df = median_year_climate[['Month', 'Day']]
            date_df['Year'] = [2000 for _ in range(len(date_df))]
            x = pd.to_datetime(date_df)

            y_temp = median_year_climate['Temp_rolling']
            y_prec = median_year_climate['Prec_cumsum_rolling']
            y_temp_sc = median_year_climate['Temp_scaled']
            y_prec_sc = median_year_climate['Prec_scaled']

            ax[i, 0].plot(x, y_temp, color='red')
            
            ax2 = ax[i, 0].twinx()
            ax2.plot(x, y_prec, color='blue')

            ax[i, 1].plot(x, y_temp_sc, c='red')
            ax[i, 1].plot(x, y_prec_sc, c='blue')
            ax[i, 1].fill_between(x, y_temp_sc, y_prec_sc, color='black', alpha=0.2)

            ax2.set_ylim([0, 350])
            
            ax[i, 0].set_ylabel('Temperature (°C)')
            ax2.set_ylabel('Precipitation (mm)')
            ax[i, 0].set_zorder(1)               # default zorder is 0 for ax1 and ax2
            ax[i, 0].patch.set_visible(False)    # prevents ax1 from hiding ax2

            locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
            formatter = mdates.ConciseDateFormatter(locator)
            ax[i, 0].xaxis.set_major_locator(locator)
            ax[i, 0].xaxis.set_major_formatter(formatter)
            ax[i, 1].xaxis.set_major_locator(locator)
            ax[i, 1].xaxis.set_major_formatter(formatter)
            ax[i, 0].set_ylim([0, 30])
            ax[i, 1].set_ylim([0, 1.1])

            ax[i, 0].set_title(f'{i+1} Class')
            ax[i, 1].set_title(f'{i+1} Class')
            ax[i, 1].set_ylabel('Climate factors (rel. units)')
            ax[i, 1].yaxis.set_label_position("right")

            ax[i, 1].yaxis.tick_right()

            loc_area = sum(abs(y_temp_sc - y_prec_sc))
            ax[i, 0].text(0.15, 0.9, f'Year {median_year}', horizontalalignment='center', verticalalignment='center', transform=ax[i, 0].transAxes)
            ax[i, 1].text(0.15, 0.9, f'Year {median_year}', horizontalalignment='center', verticalalignment='center', transform=ax[i, 1].transAxes)
            ax[i, 1].text(0.5, 0.5, f'{loc_area:.2f}', horizontalalignment='center', verticalalignment='center', transform=ax[i, 1].transAxes)
        
        return fig, ax