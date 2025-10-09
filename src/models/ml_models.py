"""
Machine Learning Models for Trading Strategies

This module provides various ML models optimized for financial time series prediction.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib


class TradingModel:
    """
    Base class for trading models with common functionality.
    """

    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        Initialize trading model
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'lightgbm', 'logistic')
            **kwargs: Additional model parameters to pass to the underlying model constructor
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.is_fitted = False
        self.model = self._create_model(**kwargs) # Instantiate model during initialization

    def _create_model(self, **kwargs) -> Any:
        """Create the specified model type"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 5),
                min_samples_leaf=kwargs.get('min_samples_leaf', 2),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
        elif self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                num_leaves=kwargs.get('num_leaves', 31),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
        elif self.model_type == 'logistic':
            return LogisticRegression(
                C=kwargs.get('C', 1.0),
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> 'TradingModel':
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            **kwargs: Additional model parameters
            
        Returns:
            Self for method chaining
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create and train model
        self.model = self._create_model(**kwargs)
        
        # Train with early stopping if validation data provided
        if X_val is not None and y_val is not None and self.model_type in ['xgboost', 'lightgbm']:
            X_val_scaled = self.scaler.transform(X_val)
            
            if self.model_type == 'xgboost':
                self.model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    verbose=False
                )
            else:  # lightgbm
                self.model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
        else:
            self.model.fit(X_train_scaled, y_train)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted probabilities for each class
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N most important features
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not available for this model")
        
        return self.feature_importance.head(top_n)

    def save(self, filepath: str) -> None:
        """Save model to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'TradingModel':
        """Load model from disk"""
        data = joblib.load(filepath)
        
        instance = cls(model_type=data['model_type'])
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_importance = data['feature_importance']
        instance.is_fitted = True
        
        return instance


class EnsembleModel:
    """
    Ensemble of multiple trading models with voting mechanism.
    """

    def __init__(self, models: list):
        """
        Initialize ensemble
        
        Args:
            models: List of TradingModel instances
        """
        self.models = models
        self.is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'EnsembleModel':
        """Train all models in the ensemble"""
        for model in self.models:
            model.fit(X_train, y_train, X_val, y_val)
        
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame, voting: str = 'hard') -> np.ndarray:
        """
        Make ensemble predictions
        
        Args:
            X: Features to predict on
            voting: 'hard' for majority voting, 'soft' for probability averaging
            
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        if voting == 'hard':
            # Majority voting
            predictions = np.array([model.predict(X) for model in self.models])
            return np.apply_along_axis(
                lambda x: np.bincount(x).argmax(),
                axis=0,
                arr=predictions
            )
        else:  # soft voting
            # Average probabilities
            probas = np.array([model.predict_proba(X) for model in self.models])
            avg_proba = np.mean(probas, axis=0)
            return np.argmax(avg_proba, axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict averaged probabilities from all models"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        probas = np.array([model.predict_proba(X) for model in self.models])
        return np.mean(probas, axis=0)
