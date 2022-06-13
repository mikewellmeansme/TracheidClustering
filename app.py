import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from itertools import product
from numpy import (
    array,
    ceil
)
from pandas import (
    DataFrame,
    read_excel,
    to_datetime
)

from normalized_tracheids import NormalizedTracheids
from clusterer import Clusterer
from utils.functions import (
    get_moving_avg,
    get_median_index
)


default_xticks = [1, 5, 10, 15, 17, 21, 26, 31]
default_xticklabels = [1, 5, 10, 15, 1, 5, 10, 15]


class Application:
    normalized_tracheids : NormalizedTracheids
    clusterer : Clusterer
    clustered_objects: DataFrame
    area_first_day: int = 121
    area_last_day: int = -92


    def __init__(self, tracheid_name,  tracheid_path, trees,
                 temperature_path, temperature_sheet, precipitation_path, precipitation_sheet) -> None:
        self.temp = read_excel(precipitation_path, sheet_name=precipitation_sheet)
        self.prec = read_excel(temperature_path, sheet_name=temperature_sheet)

        normalized_tracheids = NormalizedTracheids(tracheid_name, tracheid_path, trees)
        self.normalized_tracheids = normalized_tracheids
        self.train_clusterer()
    

    def train_clusterer(self, method: str ='A') -> None:

        if method.upper() not in 'AB':
            raise Exception(f'Wrong method given! Expected A or B, got {method}!')

        clusterer = Clusterer()
        clusterer.fit(self.normalized_tracheids.obects_for_clustering[f'Method {method}'], 4)
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
    

    def get_area_index(self):
        pass
    

    def plot_area_per_class(self):

        date_df = self.temp[['Month', 'Day']]
        date_df['Year'] = [2000 for _ in range(len(date_df))]
        x = to_datetime(date_df)[self.first_day:self.last_day]

        fig, ax = plt.subplots(nrows=4, ncols=2, dpi=300, figsize=(10, 12))
        plt.subplots_adjust(bottom=0.03, top=0.95)
        min_temp_moving_avg = get_moving_avg(self.temp).iloc[self.first_day:self.last_day].drop(columns=['Month', 'Day']).reset_index(drop=True)
        min_prec_moving_avg_cumsum = get_moving_avg(self.prec.fillna(0).cumsum()).drop(columns=['Month', 'Day']).iloc[self.first_day:self.last_day].reset_index(drop=True)


        temp_max = min_temp_moving_avg.max().max()
        prec_max = min_prec_moving_avg_cumsum.max().max()
        temp_min = min_temp_moving_avg.min().min()
        prec_min = min_prec_moving_avg_cumsum.min().min()

        scaled_min_temps = (min_temp_moving_avg - temp_min) / (temp_max - temp_min)
        scaled_min_precs =  (min_prec_moving_avg_cumsum- prec_min) / (prec_max - prec_min)


        for i in range(4):
            selected = areas_df[areas_df['Class 4'] == i]
            median_year_index = selected.apply(get_median_index, 0)['Area']
            median_year = int(selected.loc[median_year_index]['Year'])

            y_temp = min_temp_moving_avg[median_year]
            y_prec = min_prec_moving_avg_cumsum[median_year]
            y_temp_sc = scaled_min_temps[median_year]
            y_prec_sc = scaled_min_precs[median_year]

            ax[i, 0].plot(x, y_temp, color='red')
            
            ax2 = ax[i, 0].twinx()
            ax2.plot(x, y_prec, color='blue')

            ax[i, 1].plot(x, y_temp_sc, c='red')
            ax[i, 1].plot(x, y_prec_sc, c='blue')
            ax[i, 1].fill_between(x, y_temp_sc, y_prec_sc, color='black', alpha=0.2)

            ax2.set_ylim([0, 350])
            
            ax[i, 0].set_ylabel('Temperature (°C)')
            ax2.set_ylabel('Precipitation (mm)')
            ax[i, 0].set_zorder(1)  # default zorder is 0 for ax1 and ax2
            ax[i, 0].patch.set_visible(False)  # prevents ax1 from hiding ax2

            locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
            formatter = mdates.ConciseDateFormatter(locator)
            ax[i, 0].xaxis.set_major_locator(locator)
            ax[i, 0].xaxis.set_major_formatter(formatter)
            ax[i, 1].xaxis.set_major_locator(locator)
            ax[i, 1].xaxis.set_major_formatter(formatter)
            #ax.set_xlim([date(2000, 4, 1), date(2000, 10, 30)])
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