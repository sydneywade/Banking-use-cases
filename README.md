# Banking-use-cases
Machine Learning and Reinforcement Learning Portfolio for Banking Use Cases

This portfolio showcases a set of projects that implements solutions for banking clients. This portfolio includes classification, regression, clustering, dimensionality reduction, and reinforcement learning models, each of which targets a different use case.


## **1. Classification Model: Credit Card Fraud Detection**
**Dataset:** https://www.kaggle.com/mlg-ulb/creditcardfraud

**Task:** Detect fraudulent transactions from real-world credit card data collected from European cardholders. The goal is to accurately distinguish fraud (class 1) from legitimate (class 0) purchases

**Approach:** Compare Logistic Regression and XGBoost models

**Evaluation Metrics:** Precision, Recall, F1-Score, AUC-ROC

**Visualizations:** Confusion matrix, ROC curve, Class-distribution bar chart


## **2. Regression Model: Next-Month Credit Card Balance Prediction**
**Dataset:** https://www.kaggle.com/datasets/abdalrahmanelnashar/credit-card-balance-prediction?utm_source=chatgpt.com

**Task:** Predict customers' credit card balances for the next month using features such as current balance, minimum payment, credit limit, and payment history

**Approach:** Compare Random Forest and XGBoost Regression models

**Evaluation Metrics:** Log Loss, AUC-ROC, R^2

**Visualizations:** Predicted vs. actual balance scatter plot, Calibration curves


## **3. Clustering: Bank Customer Segmentation**
**Dataset:** https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation

**Task:** Cluster over 800,000 customers based on transactional and demographic data to identify meaningful marketing segements

**Approach:** Compare K-Means, DBSCAN, and Gaussian Mixture Models

**Evaluation Metrics:** Silhouette Score, Davies-Bouldin Index

**Visualizations:** 2D PCA scatter plot, Elbow plot


## **4. Dimensionality Reduction: PCA and UMAP on Credit Card Default Data**
**Dataset:** https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset?utm_source=chatgpt.com

**Task:** Compress high dimensional data and minimize the amount of information lost

**Approach:** PCA and UMAP

**Evaluation Metrics:** Explained Variance Ratio, Reconstruction Error

**Visualizations:** Scree plot, 2D component projections


## **5. Reinforcement Learning: Stock Portfolio Management**

**Dataset:** https://github.com/AI4Finance-Foundation/FinRL-Meta

**Task:** Train reinforcement learning model to optimize portfolio returns

**Approach:** Train PPO agent in a gym-style trading environment

**Evaluation Metrics:** Cumulative reward, Sharpe Ratio

**Visualizations:** Reward per episode curve, Portfolio value over time
