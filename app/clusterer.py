import numpy as np

from sklearn.cluster import KMeans
from pandas import DataFrame
from permetrics import ClusteringMetric

from collections import Counter
from typing import Optional, Dict, List


class Clusterer:
    clusters: int
    __model__: KMeans
    __number_to_name__: Optional[Dict] = None
    _metrics_: List[str] = [
        'BHI', 'XBI', 'DBI', 'BRI', 'KDI', 'DRI',
        'DI', 'CHI', 'LDRI', 'LSRI', 'SI', 'SSEI',
        'MSEI', 'DHI', 'BI', 'RSI', 'DBCVI', 'HI'
    ]

    def __init__(self) -> None:
        pass

    def fit(self, df: DataFrame, clusters: int):
        self.clusters = clusters
        data = self.__df_to_array__(df)
        model = KMeans(n_clusters=clusters, max_iter=5000, random_state=0)
        self.__model__ = model
        self.__model__.fit(data)
    
    def fit_best(self, df: DataFrame, min_clusters: int = 3, max_clusters: int = 10):
        data = self.__df_to_array__(df)
        models = []
        metric_values = {metric: [] for metric in self._metrics_}
        for clusters in range(min_clusters, max_clusters+1):
            model = KMeans(n_clusters=clusters, random_state=42, n_init=10)
            model.fit(data)
            models.append(model)
            y_pred = model.predict(data)
            evaluator = ClusteringMetric(X=data, y_pred=y_pred, decimal=5)
            for metric in metric_values:
                try:
                    val = getattr(evaluator, metric)()
                except:
                    val = np.nan
                metric_values[metric].append(val)
        
        metric_best = {}
        for metric in metric_values:
            metric_type = evaluator.SUPPORT[metric]["type"]
            metric_func = np.argmax if metric_type == 'max' else np.argmin
            metric_best[metric] = metric_func(metric_values[metric]) + min_clusters

        vote = Counter(metric_best.values())
        best_clusters = vote.most_common(1)[0][0]

        self.vote = vote
        self.metric_best = metric_best
        
        self.clusters = best_clusters
        self.__model__ = models[best_clusters - min_clusters]
        
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
