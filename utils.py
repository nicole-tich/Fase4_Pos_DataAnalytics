import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder


# Classes para pipeline de Obesidade

class MinMaxTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features=['Age', 'Height', 'Weight', 'BMI']):
        self.features = features
        self.scaler = None
    
    def fit(self, X, y=None):
        self.scaler = MinMaxScaler()
        self.scaler.fit(X[self.features])
        return self
    
    def transform(self, X):
        if set(self.features).issubset(X.columns):
            X_copy = X.copy()
            X_copy[self.features] = self.scaler.transform(X_copy[self.features])
            return X_copy
        else:
            print('Uma ou mais features não estão no DataFrame')
            return X


class BinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, binary_features=['Gender', 'family_history', 'FAVC', 'SMOKE', 'SCC']):
        self.binary_features = binary_features
        self.encoders = {}
    
    def fit(self, X, y=None):
        for feature in self.binary_features:
            if feature in X.columns:
                le = LabelEncoder()
                le.fit(X[feature])
                self.encoders[feature] = le
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for feature, encoder in self.encoders.items():
            if feature in X_copy.columns:
                X_copy[feature] = encoder.transform(X_copy[feature])
        return X_copy


class OneHotEncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, onehot_features=['CAEC', 'CALC', 'MTRANS']):
        self.onehot_features = onehot_features
        self.encoder = None
        self.feature_names = None
    
    def fit(self, X, y=None):
        if set(self.onehot_features).issubset(X.columns):
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.encoder.fit(X[self.onehot_features])
            self.feature_names = self.encoder.get_feature_names_out(self.onehot_features)
        return self
    
    def transform(self, X):
        if set(self.onehot_features).issubset(X.columns):
            X_copy = X.copy()
            
            # Aplicando one-hot encoding
            encoded = self.encoder.transform(X_copy[self.onehot_features])
            df_encoded = pd.DataFrame(encoded, columns=self.feature_names, index=X_copy.index)
            
            # Removendo as colunas originais
            X_copy = X_copy.drop(self.onehot_features, axis=1)
            
            # Concatenando com as colunas codificadas
            X_full = pd.concat([X_copy, df_encoded], axis=1)
            return X_full
        else:
            print('Uma ou mais features não estão no DataFrame')
            return X