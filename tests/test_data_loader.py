import pandas as pd
import yfinance as yf
from unittest.mock import patch
from src.utils.data_loader import DataLoader

def test_download_stock_data_success():
    with patch.object(yf.Ticker, 'history') as mock_history:
        mock_history.return_value = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [102, 103, 104],
            'Low': [99, 100, 101],
            'close': [101, 102, 103],
            'Volume': [1000, 1100, 1200]
        }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
        
        loader = DataLoader()
        data = loader.download_stock_data("AAPL", period="1d")
        
        assert not data.empty
        assert "close" in data.columns
        assert len(data) == 3
        mock_history.assert_called_once()

def test_create_target_variable():
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106]
    })
    loader = DataLoader()
    data_with_target = loader.create_target_variable(data, horizon=2, threshold=0.01)
    
    # Expected target: (Close_t+2 - Close_t) / Close_t
    # Day 0: (102 - 100) / 100 = 0.02 -> 1 (buy)
    # Day 1: (103 - 101) / 101 = 0.0198 -> 1 (buy)
    # Day 2: (104 - 102) / 102 = 0.0196 -> 1 (buy)
    # Day 3: (105 - 103) / 103 = 0.0194 -> 1 (buy)
    # Day 4: (106 - 104) / 104 = 0.0192 -> 1 (buy)
    # Day 5, 6: NaN
    
    assert isinstance(data_with_target, pd.Series)
    assert data_with_target.name == 'target'
    assert data_with_target.iloc[0] == 1 # 0.02 > 0.01
    assert data_with_target.iloc[1] == 1 # 0.0198 > 0.01
    assert data_with_target.iloc[2] == 1 # 0.0196 > 0.01
    assert data_with_target.iloc[3] == 1 # 0.0194 > 0.01
    assert data_with_target.iloc[4] == 1 # 0.0192 > 0.01
    assert pd.isna(data_with_target.iloc[5])
    assert pd.isna(data_with_target.iloc[6])

    # Test with negative return
    data_neg = pd.DataFrame({
        'close': [100, 99, 98, 97, 96, 95, 94]
    })
    data_with_target_neg = loader.create_target_variable(data_neg, horizon=2, threshold=0.01)
    # Day 0: (98 - 100) / 100 = -0.02 -> -1 (sell)
    assert data_with_target_neg.iloc[0] == -1

    # Test with neutral return
    data_neutral = pd.DataFrame({
        'close': [100, 100.5, 100.8, 100.9, 101, 101.1, 101.2]
    })
    data_with_target_neutral = loader.create_target_variable(data_neutral, horizon=2, threshold=0.01)
    # Day 0: (100.8 - 100) / 100 = 0.008 -> 0 (hold)
    assert data_with_target_neutral.iloc[0] == 0

