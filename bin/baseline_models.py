import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder

class AverageForResidueAtPosition(BaseEstimator):

    def __init__(self):
        self.train_x, self.train_y, self._test_x = None, None, None
        self._y_means = None

    def fit(self, train_x: pd.DataFrame, train_y: pd.DataFrame):
        if list(train_x.columns) != list(train_y.columns):
            raise ValueError('x columns must match y columns in this antibody setting')
        self.train_x, self.train_y = train_x, train_y
        self._y_means = self.train_y.mean(axis=0, skipna=True)

    def _avg_col(self, row):
        for column in self._test_x.columns:
            if not pd.isna(row[column]):
                row[column] = self._y_means[column]
        return row

    def predict(self, test_x: pd.DataFrame):
        self._test_x = test_x
        # copy text_x, put NaNs where are '-', 1s elsewhere
        blank = test_x.where(test_x == '-', 1).where(test_x != '-', np.nan)
        predictions = blank.apply(self._avg_col, axis=1)
        return predictions


class StatisticForSameResidueAtPosition(BaseEstimator):

    def __init__(self, statistic='mean'):
        if statistic not in ('mean', 'median'):
            raise ValueError('statistic must be "mean" or "median"')
        self.statistic = statistic
        self.train_x, self.train_y, self._test_x = None, None, None
        self._db = None

    def fit(self, train_x: pd.DataFrame, train_y: pd.DataFrame):
        if list(train_x.columns) != list(train_y.columns):
            raise ValueError('x columns must match y columns in this antibody setting')
        self.train_x, self.train_y = train_x, train_y

        self._db = dict()
        # group sasa values by (column, aminoacid) as a key
        for column in self.train_x.columns:
            self._db[column] = defaultdict(list)
            for index, value in self.train_x[column].iteritems():
                self._db[column][value].append(self.train_y.loc[index, column])

        # compute desired statistic for each (column, aminoacid) combination
        for column in self._db:
            for aa in self._db[column]:
                if self.statistic == 'mean':
                    self._db[column][aa] = np.mean(self._db[column][aa])
                elif self.statistic == 'median':
                    self._db[column][aa] = np.median(self._db[column][aa])

    def _avg_col(self, row):
        for column in self._test_x.columns:
            res = self._test_x.loc[row.name, column]
            assert column in self._db, f'train_x.columns must equal to test_x.columns - {list(self._db.keys())} vs. {list(self._test_x.columns)}'

            if not pd.isnull(row[column]) and res:
                # in test_x data there was some AA, not gap
                if type(self._db[column][res]) == list:
                    # no record in the DB, leave NaN as a result
                    pass
                else:
                    # use computed mean for given (column, residue)
                    row[column] = self._db[column][res]
        return row

    def predict(self, test_x: pd.DataFrame):
        self._test_x = test_x
        # copy text_x, put NaNs where are '-', 1s elsewhere
        blank = test_x.where(test_x == '-', 1).where(test_x != '-', np.nan)
        predictions = blank.apply(self._avg_col, axis=1)
        return predictions


class KNNWholeSequence(BaseEstimator):
    def __init__(self, n_neighbors=3):
        self.knn = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.n_neighbors = n_neighbors
        self.onehot = None
    
    def fit(self, train_x: pd.DataFrame, train_y: pd.DataFrame):
        self.onehot = OneHotEncoder(handle_unknown='ignore')
        self.onehot.fit(train_x)
        self.train_x, self.train_y = train_x, train_y
        self.train_x_oh = self.onehot.transform(train_x)
    
    def predict(self, test_x: pd.DataFrame):
        test_x_oh = self.onehot.transform(test_x)
        self.knn.fit(self.train_x_oh, self.train_y.fillna(-1))
        knn_predictions = pd.DataFrame(self.knn.predict(test_x_oh), 
                                       columns=test_x.columns, index=test_x.index)
        return knn_predictions