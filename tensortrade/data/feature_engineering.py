# Copyright 2022 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import pandas as pd
from statsmodels.tsa.stattools import adfuller


class FeatureTest:

    @classmethod
    def check_stationarity(cls, dataframe: pd.DataFrame, date_column_name: str):
        _df = dataframe.copy()
        _df[date_column_name] = pd.to_datetime(_df[date_column_name], infer_datetime_format=True)
        _df = _df.set_index([date_column_name])

        column_results = []
        for column_name in _df:

            column_test = adfuller(_df[column_name], autolag='AIC')
            test_statistic = column_test[0]
            critical_value_01 = column_test[4]['1%']
            critical_value_05 = column_test[4]['5%']
            critical_value_10 = column_test[4]['10%']

            critical_01 = False
            critical_05 = False
            critical_10 = False

            if test_statistic > critical_value_01:
                critical_01 = True

            if test_statistic > critical_value_05:
                critical_05 = True

            if test_statistic > critical_value_10:
                critical_10 = True
                print(f"{column_name} is above 10% critical")

            column_name = '_'.join(column_name.split('_')[1:]) + '_' + column_name.split('_')[0]
            column_results += [[column_name, critical_01, critical_05, critical_10]]

        df_results = pd.DataFrame(column_results, columns=['Column Name', 'Critical 1%', 'Critical 5%', 'Critical 10%'])
        df_results.sort_values('Column Name', inplace=True)
        print(df_results.head())
