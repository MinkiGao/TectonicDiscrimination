import pandas as pd

from sklearn.impute import KNNImputer

knn_imputer = KNNImputer(n_neighbors=5, weights="distance")

data_path = 'mydata'

data = pd.read_excel(data_path)

X = data[data.columns[2:]].values
X = knn_imputer.fit_transform(X)
data[data.columns[2:]] = X

