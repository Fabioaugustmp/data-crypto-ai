from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

class LRM:
    """Logistic Regression Model para classificação de direção de preço."""
    
    def __init__(self, C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, **kwargs):
        """
        Inicializa o modelo LRM.
        
        Args:
            C: Regularização
            penalty: Tipo de penalização
            solver: Algoritmo de otimização
            max_iter: Máximo de iterações
            **kwargs: Outros argumentos (filtrados)
        """
        # Filtrar argumentos depreciados
        valid_params = {
            'C': C,
            'penalty': penalty,
            'solver': solver,
            'max_iter': max_iter,
            'random_state': kwargs.get('random_state', 42)
        }
        
        # Remover multi_class se existir nos kwargs
        if 'multi_class' in kwargs:
            del kwargs['multi_class']
        
        # Adicionar outros parâmetros válidos
        for param in ['fit_intercept', 'class_weight', 'dual', 'tol', 'n_jobs', 'l1_ratio']:
            if param in kwargs:
                valid_params[param] = kwargs[param]
        
        self.regressor = LogisticRegression(**valid_params)
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        """Treina o modelo."""
        # Converter problema de regressão para classificação
        # Classificar como 1 se preço sobe, 0 se desce
        y_binary = (y > np.median(y)).astype(int)
        
        # Escalar features
        X_scaled = self.scaler.fit_transform(X)
        
        # Treinar modelo
        self.regressor.fit(X_scaled, y_binary)
        return self
        
    def predict(self, X):
        """Faz predições."""
        X_scaled = self.scaler.transform(X)
        return self.regressor.predict(X_scaled)
        
    def predict_proba(self, X):
        """Retorna probabilidades."""
        X_scaled = self.scaler.transform(X)
        return self.regressor.predict_proba(X_scaled)
        
    def get_feature_importance(self):
        """Retorna informações sobre importância das features."""
        coef = self.regressor.coef_[0] if hasattr(self.regressor, 'coef_') else None
        
        if coef is not None:
            # Importância baseada no valor absoluto dos coeficientes
            importance = np.abs(coef)
            importance_normalized = importance / np.sum(importance) if np.sum(importance) > 0 else importance
        else:
            importance_normalized = None
            
        return {
            'model_type': 'Logistic Regression',
            'n_features_in': getattr(self.regressor, 'n_features_in_', 'N/A'),
            'n_classes': len(self.regressor.classes_) if hasattr(self.regressor, 'classes_') else 'N/A',
            'classes': self.regressor.classes_.tolist() if hasattr(self.regressor, 'classes_') else 'N/A',
            'penalty': self.regressor.penalty,
            'C': self.regressor.C,
            'solver': self.regressor.solver,
            'n_iter': self.regressor.n_iter_.tolist() if hasattr(self.regressor, 'n_iter_') else 'N/A',
            'feature_importance': importance_normalized.tolist() if importance_normalized is not None else 'N/A',
            'coefficients': coef.tolist() if coef is not None else 'N/A',
            'intercept': self.regressor.intercept_.tolist() if hasattr(self.regressor, 'intercept_') else 'N/A'
        }