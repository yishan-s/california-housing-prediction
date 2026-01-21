# California Housing Price Prediction
基於 Kaggle 加州房價資料集的機器學習專案。透過該專案，我實作了從資料清洗、特徵工程、模型訓練到部署的完整流程。

## Spotlight
* **End-to-End Pipeline**：涵蓋 EDA、ETL、Modeling 與 Inference Script。
* **Feature Engineering**：使用 One-Hot Encoding 處理地理類別特徵，顯著提升模型表現。
* **Optimization**：使用 GridSearchCV 進行超參數調整，優化 Random Forest 模型。
* **Insight**：透過誤差分析發現資料集存在 50 萬美元的截斷問題。

## Tech Stack
* **Language**: Python 3.10
* **Libraries**: Pandas, Scikit-learn, Matplotlib, Seaborn
* **Model**: Random Forest

## Analysis
* **Best model**: Random Forest (經過 GridSearch 調參)
* **Testing RMSE**: 約 $48,841
* **Observations**: Median Income 是影響房價最重要的因素，其次是地理位置。

## File Structure
* `housing_analysis.ipynb`: 完整的分析與訓練過程筆記本。
* `predict.py`: 用於載入模型並進行新資料預測的腳本。
* `california_housing_rf_model.pkl`: 訓練好的模型檔。

## Implementation
1. 安裝環境:
   ```bash
   conda create -n housing_env python=3.10
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
2. Run the script:
   `python predict.py`
