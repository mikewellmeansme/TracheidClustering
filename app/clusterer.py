from sklearn.cluster import KMeans
from pandas import DataFrame
from typing import Optional, Dict


class Clusterer:
    clusters: int
    __model__: KMeans = None
    __number_to_name__: Optional[Dict] = None


    def __init__(self) -> None:
        pass


    def fit(self, df: DataFrame, clusters: int):
        self.clusters = clusters
        data = self.__df_to_array__(df)
        model = KMeans(n_clusters=clusters, max_iter=5000, random_state=0)
        self.__model__ = model
        self.__model__.fit(data)


    def predict(self, df: DataFrame) -> list:
        data = self.__df_to_array__(df)
        predictions = self.__model__.predict(data)
        result = self.convert_class_number_to_name(predictions)
        return result
    

    def change_class_names(self, number_to_name: dict[int, int]) -> None:
        self.__number_to_name__ = number_to_name
    
    
    def convert_class_number_to_name(self, class_numbers: list):
        if not self.__number_to_name__:
            return class_numbers
        class_names = [self.__number_to_name__.get(num) for num in class_numbers]
        return class_names


    @staticmethod
    def __df_to_array__(df: DataFrame):
        data = df.drop(columns=['Year']).to_numpy()
        return data
    

    
