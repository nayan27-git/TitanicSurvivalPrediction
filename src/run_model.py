import pandas as pd
from sklearn.model_selection import train_test_split
from Logistic_regression import ModelManager
from pathlib import Path

if __name__ == "__main__":
    # Get the path to the current file
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_path = BASE_DIR/"Dataset"/"train.csv"
    # Load data
    df_train = pd.read_csv(data_path)
    X = df_train.drop('Survived',axis=1)
    y = df_train['Survived']
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)

    # Initializing model pipeline
    model_map = {'logistic' : 'LogisticRegression',
                  'rf' : 'RandomForestClassifier',
                  'xgb': 'XGBClassifier'
        }
    for name, val in model_map.items():
        print(f"\n------Initializing {val}------" )
        modeler = ModelManager(model_type=name)
        modeler.train(X_train,y_train)
        modeler.evaluate(X_test, y_test)

