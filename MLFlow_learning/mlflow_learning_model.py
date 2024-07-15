import mlflow
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataset = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

mlflow.sklearn.autolog()
with mlflow.start_run():
    clf = LogisticRegression()
    clf.fit(X_train, y_train)