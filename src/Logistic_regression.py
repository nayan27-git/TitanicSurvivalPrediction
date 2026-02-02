import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, confusion_matrix, f1_score
from sklearn.base import BaseEstimator,TransformerMixin

class TitanicTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.title_mapping = {
            "Mr": "Mr",
            "Miss": "Miss",
            "Mrs": "Mrs",
            "Master": "Master",
            "Dr": "Rare",
            "Rev": "Rare",
            "Col": "Rare",
            "Major": "Rare",
            "Mlle": "Miss", # Cleaning up synonyms
            "Ms": "Miss",
            "Mme": "Mrs",
            "Don": "Rare",
            "Lady": "Rare"
        }

        self.title_means = None
        self.global_mean = None
    
    def _feature_engineering(self, df):
        df = df.copy()
        # Creating Title feature
        df['Title'] = df['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
        df['Title'] = df['Title'].map(self.title_mapping).fillna('Rare')
        # Creating family size feature
        df['Family_size'] = df['SibSp']+df['Parch']+1
        return df

    def _impute_age(self, df):
        df['Age'] = df['Age'].fillna(df['Title'].map(self.title_means))        
        df['Age'] = df['Age'].fillna(self.global_mean)
        return df

    def fit(self, X, y=None):
        temp_df = self._feature_engineering(X)
        self.title_means = temp_df.groupby('Title')['Age'].mean()
        self.global_mean = temp_df['Age'].mean()
        self.embarked_mode = X['Embarked'].mode()[0]
        return self
    
    def transform(self, X):
        # Avoid changing the original dataset
        X = X.copy()
        # Initial drops
        X = X.drop(['PassengerId','Cabin', 'Ticket'], axis=1, errors='ignore')
        # Apply transformation steps
        X = self._feature_engineering(X)
        X = self._impute_age(X)
        # Later drops
        # Inside transform()
        X['Embarked'] = X['Embarked'].fillna(self.embarked_mode)
        X = X.drop(['SibSp','Parch','Name'],axis=1, errors='ignore')
        return X


class ModelManager():

    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.hyperparameters = {}
        self.pipeline = None
        # preprocessing
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('skewed', FunctionTransformer(np.log1p), ['Fare']),
                ('unskewed', StandardScaler(), ['Pclass', 'Age', 'Family_size']),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), ['Embarked', 'Title', 'Sex'])
        ],
        remainder='passthrough'
        )    
        self.is_trained = False
        print(("Model Initialized !"))

    def _get_model(self):
        models = {'logistic' : LogisticRegression(),
                  'rf' : RandomForestClassifier(),
                  'xgb': XGBClassifier()
        }
        # return logistic regression if model key does not exist in model_type
        return models.get(self.model_type, LogisticRegression())
                
    def _build_pipeline(self):
        self.pipeline = Pipeline(steps=[
            ('cleaner_class', TitanicTransformer()),
            ('preprocessing', self.column_transformer),
            ('modeling', self._get_model()) 
        ])
        return self
    
    def train(self, X_train, y_train):
        if self.pipeline is None:
            self._build_pipeline()
        self.pipeline.fit(X_train, y_train)
        return self
    
    def evaluate(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "Log Loss": log_loss(y_test, y_pred_proba),
            "Confusion Matrix": confusion_matrix(y_test, y_pred)
        }
         
        for name, val in metrics.items():
            print(name,":", val)
        return metrics
        


