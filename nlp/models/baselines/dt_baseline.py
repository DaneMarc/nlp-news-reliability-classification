import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    for i in range(1, 12):
        train_data = pd.read_pickle(f"nlp\models\way{i}_test_doc.pkl")
        test_data = pd.read_pickle(f"nlp\models\way{i}_train_doc.pkl")
        X_train, y_train = train_data.iloc[:, -1], train_data.iloc[:, 0]
        X_train = np.vstack(X_train.to_numpy())
        y_train = y_train.to_numpy()

        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        X_test, y_test = test_data.iloc[:, -1], test_data.iloc[:, 0]
        X_test = np.vstack(X_test.to_numpy())
        y_test = y_test.to_numpy()

        print(f"Way{i}: {model.score(X_test, y_test)}")