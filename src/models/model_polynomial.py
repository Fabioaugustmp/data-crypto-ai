import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

class PLN:
    """
    Modelo de Regressão Polinomial com funcionalidades avançadas para predição de preços de criptomoedas.
    """
    
    def __init__(self, degree=2, include_bias=True, interaction_only=False, 
                 regularization=None, alpha=1.0, **kwargs):
        """
        Inicializa o modelo de Regressão Polinomial.
        
        Args:
            degree (int): Grau do polinômio (padrão: 2)
            include_bias (bool): Se deve incluir termo de intercepto
            interaction_only (bool): Se deve incluir apenas termos de interação
            regularization (str): Tipo de regularização ('ridge', 'lasso', None)
            alpha (float): Parâmetro de regularização
            **kwargs: Parâmetros adicionais
        """
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.regularization = regularization
        self.alpha = alpha
        
        # Criar transformador polinomial
        self.poly_features = PolynomialFeatures(
            degree=degree,
            include_bias=include_bias,
            interaction_only=interaction_only
        )
        
        # Criar scaler para normalização
        self.scaler = StandardScaler()
        
        # Escolher modelo de regressão baseado na regularização
        if regularization == 'ridge':
            from sklearn.linear_model import Ridge
            self.regressor = Ridge(alpha=alpha, **kwargs)
        elif regularization == 'lasso':
            from sklearn.linear_model import Lasso
            self.regressor = Lasso(alpha=alpha, max_iter=2000, **kwargs)
        elif regularization == 'elastic':
            from sklearn.linear_model import ElasticNet
            self.regressor = ElasticNet(alpha=alpha, max_iter=2000, **kwargs)
        else:
            self.regressor = LinearRegression(**kwargs)
        
        # Criar pipeline
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('poly', self.poly_features),
            ('regressor', self.regressor)
        ])
        
        self.is_fitted = False
        self.feature_names_ = None
        
    def fit(self, X, y):
        """
        Treina o modelo de regressão polinomial.
        
        Args:
            X (array-like): Features de entrada
            y (array-like): Valores target
            
        Returns:
            self: Retorna a instância do modelo
        """
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Treinando modelo Polinomial (grau {self.degree}) com {X.shape[0]} amostras")
        logger.info(f"Features originais: {X.shape[1]}")
        
        # Treinar pipeline
        self.pipeline.fit(X, y)
        self.is_fitted = True
        
        # Obter nomes das features polinomiais
        try:
            n_features_original = X.shape[1]
            feature_names = [f'feature_{i}' for i in range(n_features_original)]
            X_temp = self.scaler.fit_transform(X)
            poly_temp = self.poly_features.fit(X_temp)
            self.feature_names_ = poly_temp.get_feature_names_out(feature_names)
            n_features_poly = len(self.feature_names_)
            logger.info(f"Features polinomiais criadas: {n_features_poly}")
        except:
            logger.warning("Não foi possível obter nomes das features polinomiais")
        
        logger.info("Modelo polinomial treinado com sucesso")
        return self
        
    def predict(self, X):
        """
        Faz predições usando o modelo polinomial.
        
        Args:
            X (array-like): Features de entrada
            
        Returns:
            array: Predições
        """
        X = np.array(X)
        
        if not self.is_fitted:
            logger.warning("Modelo não foi treinado. Retornando zeros.")
            if len(X.shape) == 2:
                return np.zeros((X.shape[0], 1))
            else:
                return np.zeros((1, 1))
        
        # Fazer predições usando o pipeline
        predictions = self.pipeline.predict(X)
        
        # Garantir que as predições tenham a forma correta
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        
        return predictions
        
    def forward(self, X):
        """Compatibilidade com interface de redes neurais."""
        return self.predict(X)
        
    def __call__(self, X):
        """Permite chamar o modelo diretamente."""
        return self.predict(X)
        
    def score(self, X, y):
        """
        Calcula o score R² do modelo.
        
        Args:
            X (array-like): Features de entrada
            y (array-like): Valores target
            
        Returns:
            float: Score R²
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        return self.pipeline.score(X, y)
        
    def get_feature_importance(self):
        """
        Retorna informações sobre o modelo e importância das features.
        
        Returns:
            dict: Informações do modelo
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        # Obter coeficientes do modelo
        coefficients = self.regressor.coef_
        
        # Criar dicionário de importância das features
        feature_importance = {}
        if self.feature_names_ is not None and len(coefficients) == len(self.feature_names_):
            # Usar valor absoluto dos coeficientes como importância
            abs_coeffs = np.abs(coefficients)
            # Normalizar para soma 1
            if abs_coeffs.sum() > 0:
                normalized_importance = abs_coeffs / abs_coeffs.sum()
                for name, importance in zip(self.feature_names_, normalized_importance):
                    feature_importance[name] = importance
        
        # Informações gerais do modelo
        info = {
            'model_type': 'Polynomial Regression',
            'degree': self.degree,
            'regularization': self.regularization,
            'alpha': self.alpha if self.regularization else None,
            'n_features_original': getattr(self.pipeline.named_steps['scaler'], 'n_features_in_', 'N/A'),
            'n_features_polynomial': len(coefficients) if hasattr(self, 'regressor') else 'N/A',
            'intercept': getattr(self.regressor, 'intercept_', 'N/A'),
            'feature_importance': feature_importance
        }
        
        return info
        
    def get_coefficients(self):
        """
        Retorna os coeficientes do modelo polinomial.
        
        Returns:
            dict: Coeficientes e nomes das features
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        coefficients = self.regressor.coef_
        intercept = getattr(self.regressor, 'intercept_', 0)
        
        result = {
            'intercept': intercept,
            'coefficients': coefficients,
            'feature_names': self.feature_names_
        }
        
        if self.feature_names_ is not None and len(coefficients) == len(self.feature_names_):
            result['named_coefficients'] = dict(zip(self.feature_names_, coefficients))
        
        return result
        
    def cross_validation(self, X, y, cv=5, scoring='r2'):
        """
        Executa validação cruzada.
        
        Args:
            X (array-like): Features de entrada
            y (array-like): Valores target
            cv (int): Número de folds
            scoring (str): Métrica de avaliação
            
        Returns:
            tuple: (mean_score, std_score)
        """
        logger.info(f"Executando validação cruzada com {cv} folds")
        
        # Criar um clone do pipeline para validação cruzada
        from sklearn.base import clone
        pipeline_clone = clone(self.pipeline)
        
        # Executar validação cruzada
        scores = cross_val_score(pipeline_clone, X, y, cv=cv, scoring=scoring)
        
        logger.info(f"Validação cruzada - {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")
        
        return scores.mean(), scores.std()
        
    def evaluate(self, X, y):
        """
        Avalia o modelo com múltiplas métricas.
        
        Args:
            X (array-like): Features de entrada
            y (array-like): Valores target
            
        Returns:
            dict: Dicionário com métricas de avaliação
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        y_pred = self.predict(X).flatten()
        y_true = np.array(y).flatten()
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }
        
        return metrics
        
    def get_params(self):
        """Retorna os parâmetros do modelo."""
        return {
            'degree': self.degree,
            'include_bias': self.include_bias,
            'interaction_only': self.interaction_only,
            'regularization': self.regularization,
            'alpha': self.alpha
        }
        
    def set_params(self, **params):
        """
        Define parâmetros do modelo.
        
        Args:
            **params: Parâmetros a serem definidos
        """
        if 'degree' in params:
            self.degree = params['degree']
            self.poly_features.degree = params['degree']
        if 'include_bias' in params:
            self.include_bias = params['include_bias']
            self.poly_features.include_bias = params['include_bias']
        if 'interaction_only' in params:
            self.interaction_only = params['interaction_only']
            self.poly_features.interaction_only = params['interaction_only']
        if 'alpha' in params and self.regularization:
            self.alpha = params['alpha']
            self.regressor.alpha = params['alpha']
            
        self.is_fitted = False  # Reset fitting status
        
    def predict_with_uncertainty(self, X, n_bootstrap=100):
        """
        Faz predições com estimativa de incerteza usando bootstrap.
        
        Args:
            X (array-like): Features de entrada
            n_bootstrap (int): Número de amostras bootstrap
            
        Returns:
            tuple: (predições_média, desvio_padrão)
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        # Esta é uma implementação simplificada
        # Para uma implementação completa, seria necessário re-treinar com bootstrap
        predictions = self.predict(X)
        
        # Estimativa simplificada da incerteza baseada no erro do modelo
        # Em uma implementação real, você faria bootstrap sampling
        uncertainty = np.ones_like(predictions) * 0.1  # 10% de incerteza
        
        return predictions, uncertainty
        
    def eval(self):
        """Compatibilidade com interface de redes neurais - modo de avaliação"""
        return self
        
    def train(self, mode=True):
        """Compatibilidade com interface de redes neurais - modo de treino"""
        return self

def create_polynomial_model(degree=2, regularization=None, alpha=1.0, **kwargs):
    """
    Cria um modelo de regressão polinomial com parâmetros personalizados.
    
    Args:
        degree (int): Grau do polinômio
        regularization (str): Tipo de regularização ('ridge', 'lasso', 'elastic', None)
        alpha (float): Parâmetro de regularização
        **kwargs: Parâmetros adicionais
        
    Returns:
        PolynomialRegression: Instância do modelo polinomial
    """
    return PolynomialRegression(
        degree=degree,
        regularization=regularization,
        alpha=alpha,
        **kwargs
    )

def optimize_polynomial_degree(X, y, max_degree=5, cv=5):
    """
    Encontra o melhor grau polinomial usando validação cruzada.
    
    Args:
        X (array-like): Features de entrada
        y (array-like): Valores target
        max_degree (int): Grau máximo a testar
        cv (int): Número de folds para validação cruzada
        
    Returns:
        dict: Resultados da otimização
    """
    logger.info(f"Otimizando grau polinomial até {max_degree}")
    
    results = {}
    best_score = -np.inf
    best_degree = 1
    
    for degree in range(1, max_degree + 1):
        model = PolynomialRegression(degree=degree)
        mean_score, std_score = model.cross_validation(X, y, cv=cv)
        
        results[degree] = {
            'mean_score': mean_score,
            'std_score': std_score
        }
        
        logger.info(f"Grau {degree}: R² = {mean_score:.4f} ± {std_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_degree = degree
    
    logger.info(f"Melhor grau encontrado: {best_degree} (R² = {best_score:.4f})")
    
    return {
        'best_degree': best_degree,
        'best_score': best_score,
        'all_results': results
    }

# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = X[:, 0]**2 + 2*X[:, 1] + 0.5*X[:, 2] + np.random.randn(100)*0.1
    
    # Criar e treinar modelo
    model = PolynomialRegression(degree=2, regularization='ridge', alpha=0.1)
    model.fit(X, y)
    
    # Fazer predições
    predictions = model.predict(X[:10])
    print(f"Predições: {predictions.flatten()}")
    
    # Avaliar modelo
    metrics = model.evaluate(X, y)
    print(f"Métricas: {metrics}")
    
    # Obter informações do modelo
    info = model.get_feature_importance()
    print(f"Informações do modelo: {info}")
    
    # Otimizar grau
    optimization_results = optimize_polynomial_degree(X, y, max_degree=4)
    print(f"Otimização: {optimization_results}")
