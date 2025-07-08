import numpy as np
import pandas as pd
import joblib

dataset = pd.read_csv('C:/Users/norhan badran/Downloads/test/personality_datasert.csv')


print(dataset.isnull().sum())

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

"""from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,4])], remainder='passthrough')
X = np.array(ct.fit_transform(X))"""


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

le1 = LabelEncoder()
X[: , 1] =le1.fit_transform(X[: , 1])

le4 = LabelEncoder()
X[: , 4] =le4.fit_transform(X[: , 4])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 150, criterion = 'entropy', random_state = 0 ,max_depth=14,max_features=6)
classifier.fit(X_train, y_train)

y_pred_train = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)


from sklearn.metrics import accuracy_score
acc_train= accuracy_score(y_train , y_pred_train)
acc_test = accuracy_score(y_test , y_pred_test)
print("RF" ,acc_train)
print("RF",acc_test)

joblib.dump(classifier,"model.pkl")
joblib.dump(le, "y.pkl")
joblib.dump(le1, "le1.pkl")
joblib.dump(le4, "le4.pkl")