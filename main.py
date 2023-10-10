from typing import Literal, Tuple
import pandas as pd
from enum import Enum

class Status(Enum):
    OFF = 0
    ON = 1


df = pd.read_excel(r'data.xlsx')
OUTLIER_THRESHOLD = 1
filters_check = {
    "filter_1": {  "query": "pnl > 0", "status": Status.ON  },
    "filter_2": {  "query": "positive >= negative + 1", "status": Status.OFF  }
}   # TODO: add more checks


class Filters(Enum):
    def __new__(cls, value: int, phrase: str = ""):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.phrase = phrase
        return obj

    Drop_Percentage = 1, "Drop Percentage (%)"
    Product = 2, "Product "
    BTCUSDT_Price_Difference_Percentage = 3, "BTCUSDT Price Difference (%)"
    Symbol_Price_Difference_Percentage = 4, "Symbol Price Difference (%)"
    Specific_LS_Ratio = 5, "Specific L/S Ratio"
    General_LS_Ratio = 6, "General L/S Ratio"
    Taker_Buy_Sell_Volume = 7, "Taker Buy/Sell Volume"
    Top_Trader_Long_Short_Ratio = 8, "Top Trader Long/Short Ratio "


filters = {
    Filters.Drop_Percentage.phrase: Status.ON,
    Filters.Product.phrase: Status.ON,
    Filters.BTCUSDT_Price_Difference_Percentage.phrase: Status.ON,
    Filters.Symbol_Price_Difference_Percentage.phrase: Status.ON,
    Filters.Specific_LS_Ratio.phrase: Status.ON,
    Filters.General_LS_Ratio.phrase: Status.ON,
    Filters.Taker_Buy_Sell_Volume.phrase: Status.ON,
    Filters.Top_Trader_Long_Short_Ratio.phrase: Status.ON,
} # TODO: Make filters ON/OFF


class DataFilter:
    def __init__(self, df, filters):
        self.df = df
        self.filters = filters
        self.results_row_max = pd.Series({'Symbol': 'Maximum'})
        self.results_row_min = pd.Series({'Symbol': 'Minimum'})

    def apply_filters(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        filtered_data: pd.DataFrame = self.df
        for filter_phrase, filter_status in self.filters.items():
            if filter_status is Status.OFF: continue
            current_filter = filter_phrase
            filtered_data = self.apply_filter(current_filter, filtered_data)

        return (filtered_data, self.results_row_max, self.results_row_min)

    def apply_filter(self, current_filter, data):
        grouped_data = data.groupby(f"{current_filter}").apply(self.aggregate, filter_=current_filter)
        print("Grouped Daya:--", grouped_data)
        grouped_data = grouped_data.query(filters_check)
        self.results_row_max[current_filter] = grouped_data.index.max()
        self.results_row_min[current_filter] = grouped_data.index.min()
        filtered_data = data[data[current_filter].isin(grouped_data.index)]

        return filtered_data

    def aggregate(self, group: pd.DataFrame, filter_: str):
        """Aggregating the data such that:-
        the psitive and negative are seperated."""
        positive = group['Result'].gt(0).sum()
        negative = group['Result'].lt(0).sum()
        values = group[filter_][group['Result'].lt(0)].tolist()
        pnl = group['Result'].sum()
        pnl_percent = pnl * 100
        return pd.Series({'positive': positive, 'negative': negative,
                        'pnl': pnl_percent})


def remove_outliers(
    df: pd.DataFrame, threshold=1,
    method: Literal['robust_scaler', 'strandard_deviation'] = 'strandard_deviation'
    ) -> pd.DataFrame:

    for col in df.columns:
        if col in ['Result', 'Symbol']: continue
        """convert column to digits and remove any non-numeric"""
        df[col] = df[col].str.extract('(-?\d+\.\d+)')
        df[col] = pd.to_numeric(df[col], errors='coerce')

        if method == 'robust_scaler':
            iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
            upper_bound = df[col].quantile(0.75) + threshold * iqr
            lower_bound = df[col].quantile(0.25) - threshold * iqr
            df.drop(df[col][(df[col] > upper_bound) | (df[col] < lower_bound)].index, inplace=True)

        elif method == 'strandard_deviation':
            mean = df[col].mean()
            std = df[col].std()
            df.drop(df[col][((df[col] - mean).abs()) > (threshold * std)].index, inplace=True)

    return df


def analyze_filter(filter: Filters, data_frame: pd.DataFrame):
    pass


filters_operator = "and" # Operator will be used to join filters / queries
df = remove_outliers(df.copy(), method='robust_scaler', threshold=OUTLIER_THRESHOLD)
filters_check_exp = [filter_['query'] for filter_ in filters_check.values() if filter_['status'] is Status.ON]
filters_check = f' {filters_operator} '.join(filters_check_exp)

# Create an instance of DataFilter and apply filters
data_filter = DataFilter(df, filters)
data_filter_result, results_row_max, results_row_min = data_filter.apply_filters()

"""Adding minimum and maximum rows"""
data_filter_result.loc[len(data_filter_result)] = pd.NA
data_filter_result.loc[len(data_filter_result) + 1] = results_row_max
data_filter_result.loc[len(data_filter_result) + 2] = results_row_min

print(data_filter_result, "\n")
print(results_row_max, "\n")
print(results_row_min, "\n")
data_filter_result.to_excel('result.xlsx', index=False)