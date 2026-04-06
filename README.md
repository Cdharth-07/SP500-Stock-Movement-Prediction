# S&P 500 Stock Movement Prediction 📈🤖

## 👋 Introduction
This project is an **Applied Machine Learning** application designed to predict the daily price movement direction of the S&P 500 Index. By analyzing the interdependencies between the stock market and other global assets—such as Crude Oil, Gold, and major Forex pairs (EUR, GBP, CNY, JPY)—the model identifies patterns that signal whether the market will close higher or lower than the previous day.

## 🎯 Objective
Financial markets are highly non-linear. The goal of this project is to build a robust data pipeline that:
* Integrates multiple macro-economic datasets.
* Engineers momentum and volatility features (RSI, SMA).
* Utilizes classification algorithms to predict daily "Binary Movement" (Up/Down).

## 📊 Exploratory Data Analysis
The model begins by analyzing the distribution of features and the correlation between global assets.

### Feature Distribution
![Histograms](https://github.com/Cdharth-07/SP500-Stock-Movement-Prediction/blob/main/Images/Histograms.png?raw=true)
*Visualizing the spread of commodity prices and currency exchange rates.*

### Asset Correlation Matrix
![Heatmap](https://github.com/Cdharth-07/SP500-Stock-Movement-Prediction/blob/main/Images/Heatmap.png?raw=true)
*Identifying how the US Dollar Index and Gold prices move in relation to the S&P 500.*

## 🛠️ Tech Stack & Methodology
* **Language:** Python
* **Libraries:** Pandas (Data Cleaning), Scikit-Learn (Machine Learning), Matplotlib/Seaborn (Visualization).
* **Feature Engineering:** Relative Strength Index (RSI), Simple Moving Averages (SMA), and Asset Ratios.
* **Model:** Logistic Regression with Standardized Scaling.

## 🏆 Results
The model achieved a **Test Accuracy of 56.53%**, successfully outperforming a baseline random walk (50%). This indicates that the engineered features from global commodities and forex markets provide predictive value for stock index movements.

## 📂 Project Structure
* `Final.py`: The main execution script containing the data pipeline and ML logic.
* `Datasets/`: Historical CSV data for the S&P 500 and correlated assets.
* `images/`: Visualizations generated during the EDA phase.

---
Developed by [Sidharth Singh](https://github.com/Cdharth-07)
