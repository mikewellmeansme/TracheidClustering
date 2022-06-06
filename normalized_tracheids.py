import imp
from sqlite3 import DatabaseError
from pandas import (
    ExcelFile,
    DataFrame,
    Series,
    concat
)
from dataclasses import dataclass

from utils.functions import get_normalized_list
from utils.correlation import dropna_spearmanr


@dataclass
class NormalizedTracheids:

    name: str
    file_path: str
    trees: list
    norm_to: int = 15
    year_threshold: int = 3
    normalized_df : DataFrame = None
    mean_objects_years: dict[int, Series] = None
    mean_objects_trees: dict[int, Series] = None
    mean_object_global: Series = None
    obects_for_clustering: dict[str, DataFrame] = None
    methods_comparison: DataFrame = None


    def __post_init__(self) -> None:
        self.normalized_df = self._get_normalized_df()

        self.mean_objects_years = self._get_mean_object_years()
        self.mean_objects_trees = self._get_mean_object_trees()
        self.mean_object_global = self._get_mean_object_global()

        self.obects_for_clustering = self._get_obects_for_clustering()

        self.methods_comparison = self._get_methods_comparison()
    

    def to_csv(self, save_path: str) -> None:
        self.normalized_df.to_csv(f'{save_path}/{self.name}_normalized_df.csv', index=False)
        self.obects_for_clustering['Method A'].to_csv(f'{save_path}/{self.name}_obects_for_clustering_A.csv', index=False)
        self.obects_for_clustering['Method B'].to_csv(f'{save_path}/{self.name}_obects_for_clustering_B.csv', index=False)
        self.methods_comparison.to_csv(f'{save_path}/{self.name}_methods_comparison.csv', index=False)
    

    def to_excel(self, save_path: str) -> None:
        self.normalized_df.to_excel(f'{save_path}/{self.name}_normalized_df.xlsx', index=False)
        self.obects_for_clustering['Method A'].to_excel(f'{save_path}/{self.name}_obects_for_clustering_A.xlsx', index=False)
        self.obects_for_clustering['Method B'].to_excel(f'{save_path}/{self.name}_obects_for_clustering_B.xlsx', index=False)
        self.methods_comparison.to_excel(f'{save_path}/{self.name}_methods_comparison.xlsx', index=False)


    def __get_columns(self, tree_column:bool = True) -> dict[int, str]:
        i = int(tree_column)

        columns = {_:f'D{_-i}' if _ < self.norm_to + 1 + i else f'CWT{_ - self.norm_to-i}' for _ in  range(1+i, self.norm_to * 2 + 1 + i)}
        columns[0] = 'Tree' if tree_column else 'Year'

        if tree_column:
            columns[1] = 'Year'

        return columns
    

    def _get_normalized_df(self) -> DataFrame:

        xlsx_file = ExcelFile(self.file_path)

        columns = self.__get_columns()
        
        dataframes = []

        for tree in self.trees:
            df = xlsx_file.parse(tree)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df = df.dropna(axis=0)

            norm_traches = dict()
            for year in set(df['Год']):
                norm_traches[int(year)] = [tree, int(year)] +\
                    get_normalized_list(list(df[df['Год']==year]['Dmean']), self.norm_to) +\
                    get_normalized_list(list(df[df['Год']==year]['CWTmean']), self.norm_to)
            
            dataframes += [DataFrame(norm_traches).transpose().rename(columns=columns).reset_index(drop=True)]

        normalized_df = concat(dataframes).reset_index(drop=True)

        return normalized_df


    def _get_mean_object_years(self) -> dict[int, Series]:
        mean_objects_years = dict()

        for year in set(self.normalized_df['Year']):
            temp_data = self.normalized_df[self.normalized_df['Year']==year]
            temp_data = temp_data.drop(columns=['Year', 'Tree'])
            if len(temp_data) > self.year_threshold:
                mean_objects_years[year] = temp_data.mean()
        
        return mean_objects_years


    def _get_mean_object_trees(self) -> dict[int, Series]:
        mean_objects_trees = dict()

        for tree in set(self.normalized_df['Tree']):
            temp_data = self.normalized_df[self.normalized_df['Tree']==tree]
            temp_data = temp_data.drop(columns=['Year', 'Tree'])
            mean_objects_trees[tree] = temp_data.mean()
        
        return mean_objects_trees


    def _get_mean_object_global(self) -> Series:
        temp_data = self.normalized_df.drop(columns=['Year', 'Tree'])
        mean_object_global = temp_data.mean()

        return mean_object_global
    

    def _get_obects_for_clustering(self) -> dict[str, DataFrame]:

        obects_for_clustering = {
            'Method A': self._method_A(),
            'Method B': self._method_B(),
        }

        return obects_for_clustering


    def _method_A(self) -> DataFrame:
        objects_method_A = []

        columns = self.__get_columns(False)

        for year, mean_obj in self.mean_objects_years.items():
            objects_method_A += [[year] + list(mean_obj/self.mean_object_global)]

        objects_method_A = DataFrame(objects_method_A).rename(columns=columns)

        return objects_method_A


    def _method_B(self) -> DataFrame:

        objects_method_B = dict()

        columns = self.__get_columns()
        
        df = []

        for _, row in self.normalized_df.iterrows():
            df += [[row[0], row[1]] + list(row[2:] / self.mean_objects_trees[row[0]])]

        df = DataFrame(df).rename(columns=columns)

        for year in set(self.normalized_df['Year']):
            temp_data = df[df['Year']==year]
            temp_data = temp_data.drop(columns=['Year', 'Tree'])

            if len(temp_data) > self.year_threshold:
                objects_method_B[year] = temp_data.mean()

        objects_method_B = DataFrame(objects_method_B).transpose()
        objects_method_B = objects_method_B.reset_index().rename(columns={'index':'Year'})

        return objects_method_B
    

    def _get_methods_comparison(self) -> DataFrame:
        r_coeffs = []
        p_values = []
        columns = self.obects_for_clustering['Method A'].drop(columns=['Year']).columns
        for column in columns:
            _c, _p = dropna_spearmanr(self.obects_for_clustering['Method A'][column], self.obects_for_clustering['Method B'][column])
            r_coeffs.append(_c)
            p_values.append(_p)

        spearman_corr_df = DataFrame({'Feature':columns, 'Spearman R': r_coeffs, 'P-value': p_values})
        return spearman_corr_df
