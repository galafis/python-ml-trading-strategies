import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.ml_models import EnsembleModel, TradingModel

# Mock data for testing
X_train_data = pd.DataFrame(
    np.random.rand(100, 10), columns=[f"feature_{i}" for i in range(10)]
)
y_train_data = pd.Series(np.random.randint(0, 2, 100))
X_test_data = pd.DataFrame(
    np.random.rand(50, 10), columns=[f"feature_{i}" for i in range(10)]
)


def test_trading_model_init_valid_types():
    model_rf = TradingModel(model_type="random_forest")
    assert model_rf.model_type == "random_forest"
    assert isinstance(model_rf.model, RandomForestClassifier)
    assert isinstance(model_rf.scaler, StandardScaler)
    assert not model_rf.is_fitted

    model_xgb = TradingModel(model_type="xgboost")
    assert model_xgb.model_type == "xgboost"
    assert isinstance(model_xgb.model, xgb.XGBClassifier)

    model_lgbm = TradingModel(model_type="lightgbm")
    assert model_lgbm.model_type == "lightgbm"
    assert isinstance(model_lgbm.model, lgb.LGBMClassifier)

    model_lr = TradingModel(model_type="logistic")
    assert model_lr.model_type == "logistic"
    assert isinstance(model_lr.model, LogisticRegression)


def test_trading_model_init_invalid_type():
    with pytest.raises(ValueError) as excinfo:
        TradingModel(model_type="invalid")
    assert "Unknown model type: invalid" in str(excinfo.value)


def test_trading_model_fit_predict():
    model = TradingModel(model_type="random_forest")
    model.fit(X_train_data, y_train_data)
    assert model.is_fitted
    assert model.model is not None

    predictions = model.predict(X_test_data)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X_test_data)


def test_trading_model_predict_proba():
    model = TradingModel(model_type="random_forest")
    model.fit(X_train_data, y_train_data)
    probabilities = model.predict_proba(X_test_data)
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (len(X_test_data), 2)  # Binary classification


def test_trading_model_feature_importance():
    model = TradingModel(model_type="random_forest")
    model.fit(X_train_data, y_train_data)
    importance = model.get_feature_importance(top_n=5)
    assert isinstance(importance, pd.DataFrame)
    assert len(importance) <= 5
    assert "feature" in importance.columns
    assert "importance" in importance.columns


def test_trading_model_save_load(tmp_path):
    model = TradingModel(model_type="random_forest")
    model.fit(X_train_data, y_train_data)
    filepath = tmp_path / "test_model.joblib"
    model.save(filepath)

    loaded_model = TradingModel.load(filepath)
    assert loaded_model.is_fitted
    assert loaded_model.model_type == "random_forest"
    # Check if predictions are the same
    np.testing.assert_array_equal(
        model.predict(X_test_data), loaded_model.predict(X_test_data)
    )


def test_ensemble_model_init():
    model1 = TradingModel(model_type="random_forest")
    model2 = TradingModel(model_type="xgboost")
    ensemble = EnsembleModel([model1, model2])
    assert len(ensemble.models) == 2
    assert not ensemble.is_fitted


def test_ensemble_model_fit_predict_hard_voting():
    model1 = TradingModel(model_type="random_forest")
    model2 = TradingModel(model_type="xgboost")
    ensemble = EnsembleModel([model1, model2])
    ensemble.fit(X_train_data, y_train_data)
    assert ensemble.is_fitted

    predictions = ensemble.predict(X_test_data, voting="hard")
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X_test_data)


def test_ensemble_model_predict_soft_voting():
    model1 = TradingModel(model_type="random_forest")
    model2 = TradingModel(model_type="xgboost")
    ensemble = EnsembleModel([model1, model2])
    ensemble.fit(X_train_data, y_train_data)

    predictions = ensemble.predict(X_test_data, voting="soft")
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X_test_data)


def test_ensemble_model_predict_proba():
    model1 = TradingModel(model_type="random_forest")
    model2 = TradingModel(model_type="xgboost")
    ensemble = EnsembleModel([model1, model2])
    ensemble.fit(X_train_data, y_train_data)

    probabilities = ensemble.predict_proba(X_test_data)
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (len(X_test_data), 2)  # Binary classification
