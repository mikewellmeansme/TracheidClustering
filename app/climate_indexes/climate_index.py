from pandas import DataFrame, read_csv


class ClimateIndex:
    name: str
    climate_index: DataFrame

    def __init__(self, name: str, path: str) -> None:
        self.name = name
        self.climate_index = read_csv(path)
