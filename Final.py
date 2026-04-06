import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def load_and_preprocess_data(file_paths):
    dfs = {}
    for key, path in file_paths.items():
        if os.path.exists(path):
            print(f"✅ Loading: {path}")
            # Load everything as strings initially to prevent type errors
            dfs[key] = pd.read_csv(path, dtype=str)
        else:
            print(f"❌ Error: File NOT found at {path}")
            continue

    # Fill missing values
    for key in dfs:
        dfs[key] = dfs[key].ffill().bfill()
        # Convert Date safely
        dfs[key]['Date'] = pd.to_datetime(dfs[key]['Date'], errors='coerce')
        dfs[key] = dfs[key].dropna(subset=['Date']).set_index('Date')

    return dfs

def merge_datasets(dfs):
    if 'sp500_df' not in dfs: return None
    combined_df = dfs['sp500_df']
    other_keys = ['gold_df', 'crudeoil_df', 'eur_df', 'gbp_df', 'cny_df', 'jpy_df', 'usidx_df']
    for key in other_keys:
        if key in dfs:
            combined_df = combined_df.join(dfs[key], how='inner', rsuffix=f'_{key}')
    return combined_df

def clean_all_columns(df):
    """Aggressively forces every column to be a float, stripping all non-numeric characters."""
    print("🧹 Cleaning data types...")
    for col in df.columns:
        # Convert to string, strip commas/percents, then force to numeric
        df[col] = (
            df[col].astype(str)
            .str.replace(r'[^\d\.\-eE]', '', regex=True) # Remove everything except digits, dots, minus, and E
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Final fill for any conversion failures
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def perform_eda(df):
    print("\n📈 Histograms... (Close window to continue)")
    df.hist(figsize=(15, 10))
    plt.tight_layout()
    plt.show()

    print("🔥 Heatmap... (Close window to continue)")
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def feature_engineering(df):
    print("🛠️ Engineering features...")
    # Ensure Price is float
    df['Price'] = df['Price'].astype(float)
    
    df['Momentum_10d'] = df['Price'] - df['Price'].shift(10)
    df['SMA_10'] = df['Price'].rolling(window=10).mean()
    df['SMA_20'] = df['Price'].rolling(window=20).mean()
    
    if 'Price_gold_df' in df.columns:
        df['Gold_to_SP500'] = df['Price_gold_df'].astype(float) / df['Price']

    delta = df['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def main():
    data_dir = "Datasets"
    file_paths = {
        "sp500_df": os.path.join(data_dir, "S&P 500 Futures 2016 24.csv"),
        "crudeoil_df": os.path.join(data_dir, "Commodities Crude Oil WTI Futures 2016 24.csv"),
        "gold_df": os.path.join(data_dir, "Commodities Gold Futures 2016 24.csv"),
        "eur_df": os.path.join(data_dir, "Forex EUR_USD 2016 24.csv"),
        "gbp_df": os.path.join(data_dir, "Forex GBP_USD 2016 24.csv"),
        "cny_df": os.path.join(data_dir, "Forex USD_CNY 2016 24.csv"),
        "jpy_df": os.path.join(data_dir, "Forex USD_JPY 2016 24.csv"),
        "usidx_df": os.path.join(data_dir, "US Dollar Index 2016 24.csv"),
    }

    dfs = load_and_preprocess_data(file_paths)
    combined_df = merge_datasets(dfs)

    if combined_df is not None:
        # 1. Clean first
        combined_df = clean_all_columns(combined_df)
        
        # 2. EDA second
        perform_eda(combined_df)
        
        # 3. Features third
        combined_df = feature_engineering(combined_df)
        
        # 4. Target & Modeling
        combined_df['Binary Movement'] = 0
        combined_df.loc[combined_df['Price'].diff() > 0, 'Binary Movement'] = 1
        combined_df.loc[combined_df['Price'].diff() < 0, 'Binary Movement'] = -1
        
        drop_patterns = ['Open', 'High', 'Low', 'Vol.', 'Change %', 'Price']
        cols_to_drop = [col for col in combined_df.columns if any(p in col for p in drop_patterns) and col != 'Binary Movement']
        combined_df.drop(columns=cols_to_drop, inplace=True)
        combined_df.dropna(inplace=True)

        X = combined_df.drop(['Binary Movement'], axis=1)
        y = combined_df['Binary Movement']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("\n🤖 Training Models...")
        pipe = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=1000))])
        pipe.fit(X_train, y_train)
        print(f"Logistic Regression Accuracy: {accuracy_score(y_test, pipe.predict(X_test)):.4f}")
    else:
        print("❌ Merge failed.")

if __name__ == "__main__":
    main()