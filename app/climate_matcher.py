from re import L
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from scipy.stats import mstats

from datetime import date
from numpy import arange
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
            climate_indexes_paths: dict[str, str]
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
        

    def __load_climate_index__(self, path, clustered_objects):
        df = pd.read_csv(path)
        result = df.merge(clustered_objects[['Year', 'Class']], on='Year', how='left')
        return result
    

    def change_class_names(self, clusterer: Clusterer) -> None:
        for index in self.climate_indexes:
            self.climate_indexes[index]['Class'] = clusterer.convert_class_number_to_name(
                self.climate_indexes[index]['Class']
            )
    

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


    def __get_area_index__(self, clustered_objects: pd.DataFrame) -> None:
        r"""
        Function for calculating Area Index for the given climate data
        """
        self.__get_cut_climate__()
        cut_сlimate_df = self.cut_сlimate
        area_df = cut_сlimate_df[['Year', 'Temp_prec_difference']].groupby('Year').sum().reset_index()
        area_df = area_df.rename(columns={'Temp_prec_difference': 'Area'})

        area_df = area_df.merge(clustered_objects[['Year', 'Class']], on='Year', how='left')
        self.area = area_df
    

    def plot_area_per_class(
            self,
            clustered_objects: pd.DataFrame,
            xlim: list = [date(2000, 4, 20), date(2000, 10, 10)],
            temp_ylim: list = [0, 30],
            prec_ylim: list = [0,350]
        ) -> tuple:

        r"""
        Params:
            clustered_objects: 
            xlim: 
            temp_ylim:
            prec_ylim:
        """

        nclasses = self.__get_nclasses__()

        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)

        fig, ax = plt.subplots(nrows=nclasses, ncols=2, dpi=300, figsize=(10, 3 * nclasses))
        plt.subplots_adjust(bottom=0.03, top=0.95)

        for i in range(nclasses):
            ax2 = ax[i, 0].twinx()

            ax[i, 0].set_ylabel('Temperature (°C)')
            ax2.set_ylabel('Precipitation (mm)')
            ax[i, 0].set_zorder(1)               # default zorder is 0 for ax1 and ax2
            ax[i, 0].patch.set_visible(False)    # prevents ax1 from hiding ax2

            ax[i, 0].set_xlim(xlim)
            ax[i, 1].set_xlim(xlim)
            ax[i, 0].set_ylim(temp_ylim)
            ax2.set_ylim(prec_ylim)
            ax[i, 1].set_ylim([0, 1.1])

            ax[i, 0].set_title(f'{i+1} Class')
            ax[i, 1].set_title(f'{i+1} Class')
            ax[i, 1].set_ylabel('Climate factors (rel. units)')
            ax[i, 1].yaxis.set_label_position("right")

            ax[i, 1].yaxis.tick_right()
            
            ax2.xaxis.set_major_locator(locator)
            ax2.xaxis.set_major_formatter(formatter)
            ax[i, 0].xaxis.set_major_locator(locator)
            ax[i, 0].xaxis.set_major_formatter(formatter)
            ax[i, 1].xaxis.set_major_locator(locator)
            ax[i, 1].xaxis.set_major_formatter(formatter)

            selected = SuperbDataFrame(self.area[self.area['Class'] == i])

            if len(selected) == 0:
                continue
            
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
            
            ax2.plot(x, y_prec, color='blue')

            ax[i, 1].plot(x, y_temp_sc, c='red')
            ax[i, 1].plot(x, y_prec_sc, c='blue')
            ax[i, 1].fill_between(x, y_temp_sc, y_prec_sc, color='black', alpha=0.2)
            
            loc_area = sum(abs(y_temp_sc - y_prec_sc))
            ax[i, 0].text(0.15, 0.9, f'Year {median_year}', horizontalalignment='center', verticalalignment='center', transform=ax[i, 0].transAxes)
            ax[i, 1].text(0.15, 0.9, f'Year {median_year}', horizontalalignment='center', verticalalignment='center', transform=ax[i, 1].transAxes)
            ax[i, 1].text(0.5, 0.5, f'{loc_area:.2f}', horizontalalignment='center', verticalalignment='center', transform=ax[i, 1].transAxes)
        
        return fig, ax
    

    def plot_mean_climate_pre_class(
            self,
            clustered_objects: pd.DataFrame,
            xlim: list = [date(2000, 4, 20), date(2000, 10, 10)],
            temp_ylim: list = [0, 30],
            prec_ylim: list = [0,350]
        ) -> tuple:

        r"""
        Params:
            clustered_objects:
            xlim:
            temp_ylim:
            prec_ylim: 
        """

        nclasses = len(set(clustered_objects['Class']))

        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)

        fig, ax = plt.subplots(nrows=nclasses, ncols=2, dpi=300, figsize=(10, 3 * nclasses))
        plt.subplots_adjust(bottom=0.03, top=0.95)
        return fig, ax
    

    def boxplot_climate(
            self,
            clustered_objects: pd.DataFrame,
            ylim: list = None,
            column: str = 'Temperature',
            classes: list = None,
            start_month = growth_season_start_month,
            end_month = growth_season_end_month,
            start_day = 1,
            end_day = 31,
            moving_avg_window: int = None
        ) -> tuple:

        r"""
        Params:
            clustered_objects:
            ylim:
            column:
            classes:
            start_month:
            end_month:
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
            moving_avg_window
        )

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), dpi=200)
        ax.boxplot(totals)
        title = f"{'Mean' if column == 'Temperature' else 'Total'} {column}"
        ax.set_title(title)
        ax.set_xticklabels([cl + 1 for cl in classes])

        if ylim:
            ax.set_ylim(ylim)

        return fig, ax
    

    def get_climate_kruskalwallis(
            self,
            clustered_objects: pd.DataFrame,
            column: str = 'Temperature',
            classes: list = None,
            start_month = growth_season_start_month,
            end_month = growth_season_end_month,
            start_day = 1,
            end_day = 31,
            moving_avg_window: int = None
        ) -> tuple[float, float]:
        
        r"""
        Params:
            clustered_objects:
            ylim:
            column:
            classes:
            start_month:
            end_month:
        """

        totals = self.__get_total_climate__(
            clustered_objects,
            column,
            classes,
            start_month,
            end_month,
            start_day,
            end_day,
            moving_avg_window
        )

        s, p = mstats.kruskalwallis(*totals)

        return s, p
    

    def __get_total_climate__(
            self,
            clustered_objects: pd.DataFrame,
            column: str,
            classes: list,
            start_month: int,
            end_month: int,
            start_day: int,
            end_day: int,
            moving_avg_window: int = None

        ) -> list[list[float]]:
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
                df = self.climate[(self.climate['Year'] == year) &
                 (start_month <= self.climate['Month']) &
                 (self.climate['Month'] <= end_month)].dropna()[column]

                if len(df) > 0:
                    res = df.mean() if column == 'Temperature' else df.sum()
                    values.append(res)
            totals.append(values)
        
        return totals


    def boxplot_climate_index(
            self,
            index='PDSI', 
            prev: bool = False,
            month: str = None,
            classes: list = None,
            ylims: list = None
        ) -> tuple:

        r"""
        Params:
            index: PDSI, Area, SPEI
            prev: 
            month: 
            classes: 
            ylims: 
        """

        self.__validate_inedx__(index)
        classes = classes if classes else self.__get_classes__()
        groups = self.__get_classes_rows__(self.climate_indexes[index])

        if prev:
            df = self.__get_shifted_df__(self.climate_indexes[index])
        else:
            df = self.climate_indexes[index]

        column = month if month else index

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), dpi=200)
        ax.boxplot([list(df.loc[groups[j]][column]) for j in classes])
        title = f"{index} {month if month else ''}{' prev' if prev else ''}"
        ax.set_title(title)
        ax.set_xticklabels([cl + 1 for cl in classes])

        if ylims:
            ax.set_ylim(ylims)

        return fig, ax


    def get_climate_index_kruskalwallis(
            self,
            index='PDSI',
            prev: bool = False,
            month: str = None,
            classes: list = None
        ) -> tuple[float, float]:

        r"""
        Params:
            index: PDSI, Area, SPEI
            prev: 
            month:
            classes:
        """

        self.__validate_inedx__(index)
        classes = classes if classes else self.__get_classes__()
        groups = self.__get_classes_rows__(self.climate_indexes[index])

        if prev:
            df = self.__get_shifted_df__(self.climate_indexes[index])
        else:
            df = self.climate_indexes[index]

        column = month if month else index

        s, p = mstats.kruskalwallis(
                *[list(df.loc[groups[j], column]) for j in classes]
            )

        return s, p
    

    def get_climate_comparison(
            self,
            clustered_objects: pd.DataFrame,
            start_month = growth_season_start_month,
            end_month = growth_season_end_month
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
            ylims: dict = {
                'std': [0, 2],
                'PDSI': [-9, 9],
                'SPEI': [-1.5, 1.5],
                'Area': [20, 80],
                'Class': [0.5,4.5]
            },
            yticks: dict = {
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
    

    def __get_classes__(self) -> set[int]:
        classes = set(self.area['Class'])
        return classes


    def __get_nclasses__(self) -> int:
        classes = self.__get_classes__()
        nclasses = len(classes)
        return nclasses
    

    @staticmethod
    def __get_shifted_df__(df) -> pd.DataFrame:
        df_copy = df.copy()
        columns = [col for col in df_copy.columns if col not in ['Year', 'Class']]
        for column in columns:
            df_copy[column] = df_copy[column].shift(1)
        return df_copy.dropna()


    @staticmethod
    def __get_classes_rows__(df:pd.DataFrame, ) -> dict[int, list]:
        groups = df.groupby('Class').groups
        return groups
