import joblib
import numpy as np
import pandas as pd

def predict_house_value(features):
    """
    載入模型並進行預測
    """
    # 1. 載入模型
    model = joblib.load('california_housing_rf_model.pkl')
    
    # 2. 進行預測
    prediction = model.predict(features)
    return prediction[0]

if __name__ == "__main__":
    # --- 模擬使用者輸入 ---
    # 這裡我們手動輸入一筆測試資料 (對應 X_train 的欄位順序)
    # 假設這是一間：經度-122, 緯度37, 屋齡20年, 房間數1000... 收入中位數 8.5 (很高!) ... 且是 NEAR BAY
        
    print("loadinig and predicting...")
    
    model = joblib.load('california_housing_rf_model.pkl')
    expected_features = model.n_features_in_
    
    # 創建假資料 (全為 0)
    sample_input = np.zeros((1, expected_features))
    
    sample_input[0, 7] = 8.0  # 設定收入很高 (Median Income = 8.0)
    
    result = predict_house_value(sample_input)
    
    print("-" * 30)
    print(f"預測結果：這間房子的預估價值為 ${result:,.2f}")
    print("-" * 30)