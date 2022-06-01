from pandas import (
    ExcelFile,
    DataFrame,
    Series,
    concat
)

from utils.functions import get_normalized_list


class NormalizedTracheids:

    file_path: str
    trees: list
    norm_to: int
    normalized_df : DataFrame
    mean_objects_years: dict[str, Series]
    mean_objects_trees: dict[str, Series]
    mean_object_global: Series
    obects_for_clustering: dict[str, DataFrame]


    def __init__(self, file_path, trees, norm_to=15, year_threshold=3):
        self.file_path = file_path
        self.trees = trees
        self.norm_to = norm_to
        self.year_threshold = year_threshold

        xlsx_file = ExcelFile(file_path)

        columns = self.__get_columns()
        
        dataframes = []

        for tree in trees:
            df = xlsx_file.parse(tree)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df = df.dropna(axis=0)

            norm_traches = dict()
            for year in set(df['Год']):
                norm_traches[int(year)] = [tree, int(year)] +\
                    get_normalized_list(list(df[df['Год']==year]['Dmean']), norm_to) +\
                    get_normalized_list(list(df[df['Год']==year]['CWTmean']), norm_to)
            
            dataframes += [DataFrame(norm_traches).transpose().rename(columns=columns).reset_index(drop=True)]

        normalized_df = concat(dataframes).reset_index(drop=True)

        self.normalized_df = normalized_df

        self.mean_objects_years = self._get_mean_object_years()
        self.mean_objects_trees = self._get_mean_object_trees()
        self.mean_object_global = self._get_mean_object_global()

        self.obects_for_clustering = self._get_obects_for_clustering()


    def _get_mean_object_years(self):
        mean_objects_years = dict()

        for year in set(self.normalized_df['Year']):
            temp_data = self.normalized_df[self.normalized_df['Year']==year]
            temp_data = temp_data.drop(columns=['Year', 'Tree'])
            if len(temp_data) > self.year_threshold:
                mean_objects_years[year] = temp_data.mean()#[1:]
        
        return mean_objects_years


    def _get_mean_object_trees(self):
        mean_objects_trees = dict()

        for tree in set(self.normalized_df['Tree']):
            temp_data = self.normalized_df[self.normalized_df['Tree']==tree]
            temp_data = temp_data.drop(columns=['Year', 'Tree'])
            mean_objects_trees[tree] = temp_data.mean()#[1:]
        
        return mean_objects_trees


    def _get_mean_object_global(self):
        temp_data = self.normalized_df.drop(columns=['Year', 'Tree'])
        mean_object_global = temp_data.mean()#[1:]

        return mean_object_global
    

    def _get_obects_for_clustering(self):

        obects_for_clustering = {
            'Method A': self._method_A(),
            'Method B': self._method_B(),
        }

        return obects_for_clustering


    def _method_A(self):
        objects_method_A = []

        columns = self.__get_columns(False)

        for year, mean_obj in self.mean_objects_years.items():
            objects_method_A += [[year] + list(mean_obj/self.mean_object_global)]

        objects_method_A = DataFrame(objects_method_A).rename(columns=columns)

        return objects_method_A


    def _method_B(self):

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
                objects_method_B[year] = temp_data.mean()#[1:]

        objects_method_B = DataFrame(objects_method_B).transpose()
        objects_method_B = objects_method_B.reset_index().rename(columns={'index':'Year'})

        return objects_method_B
    

    def __get_columns(self, tree_column=True):
        i = int(tree_column)

        columns = {_:f'D{_-i}' if _ < self.norm_to + 1 else f'CWT{_ - self.norm_to-i}' for _ in  range(1+i, self.norm_to * 2 + 1 + i)}
        columns[0] = 'Year' if not tree_column else 'Tree'

        if tree_column:
            columns[1] = 'Year'

        return columns
