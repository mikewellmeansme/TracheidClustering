from pandas import DataFrame, read_csv


class ClimateIndex:
    name: str
    data: DataFrame

    def __init__(self, name: str, path: str) -> None:
        self.name = name
        self.data = read_csv(path)
