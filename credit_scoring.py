import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor
import os

try:
    import ujson as fast_json
except ImportError:
    import json as fast_json

def flatten_json(y):
    out = {}
    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], f"{name}{a}_")
        elif isinstance(x, list):
            for i, a in enumerate(x):
                flatten(a, f"{name}{i}_")
        else:
            out[name[:-1]] = x
    flatten(y)
    return out

def load_json_file(path):
    try:
        with open(path, 'r') as f:
            data = fast_json.load(f)
            if isinstance(data, dict):
                first_key = next(iter(data), None)
                if first_key and isinstance(data[first_key], list):
                    data = data[first_key]
                else:
                    print(f"⚠ Unexpected dict structure in {path}. Skipping.")
                    return pd.DataFrame()
            if not isinstance(data, list):
                print(f"⚠ Unsupported JSON format in {path}. Skipping.")
                return pd.DataFrame()
            flat_data = [flatten_json(d) for d in data if isinstance(d, dict)]
            if flat_data:
                print(f"📂 {path} — Loaded {len(flat_data)} records.")
                print(f"🔑 Columns (first record): {list(flat_data[0].keys())}")
            else:
                print(f"⚠ No usable records in {path}")
            return pd.DataFrame(flat_data)
    except Exception as e:
        print(f"⚠ Error reading {path}: {e}")
        return pd.DataFrame()

def load_all_json(filepaths):
    with ThreadPoolExecutor() as executor:
        dfs = list(executor.map(load_json_file, filepaths))
    if not dfs:
        print("⚠ No data was loaded from the provided file paths.")
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def preprocess(df):
    df = df.dropna(axis=1, how='all').dropna()
    ts_cols = [col for col in df.columns if any(k in col.lower() for k in ['time', 'timestamp', 'block'])]
    if ts_cols:
        print(f"🕒 Detected timestamp columns: {ts_cols}")
        df.rename(columns={ts_cols[0]: 'timestamp'}, inplace=True)
        try:
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            if df['timestamp'].isna().all():
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        except Exception as e:
            print(f"⚠ Error parsing timestamps: {e}")
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        print(f"🔑 Timestamp column preview:\n{df['timestamp'].head()}")
        df.dropna(subset=['timestamp'], inplace=True)
    else:
        raise Exception("No timestamp column detected. Available columns: " + str(df.columns.tolist()))
    return df

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        print("⚠ The dataframe is empty after loading.")
        return df

    print("\n🔎 Checking the first few rows of the dataframe:")
    print(df.head())

    if 'account_id' not in df.columns:
        print("⚠ Missing 'account_id' column!")
        return pd.DataFrame()

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    print(f"🔑 Converted 'timestamp' to datetime. Null values: {df['timestamp'].isna().sum()}")
    df = df.dropna(subset=['timestamp'])

    df = df.sort_values(by=['account_id', 'timestamp'])

    df['time_diff'] = df.groupby('account_id')['timestamp'].diff().dt.total_seconds()
    df['amountUSD'] = pd.to_numeric(df['amountUSD'], errors='coerce')

    # New features
    df['is_borrow'] = df['amountUSD'] < 0
    df['is_repay'] = df['amountUSD'] > 0

    agg_df = df.groupby('account_id', as_index=False).agg({
        'amountUSD': ['sum', 'mean', 'std'],
        'hash': 'count',
        'time_diff': ['mean', 'std'],
        'asset_symbol': pd.Series.nunique,
        'is_borrow': 'sum',
        'is_repay': 'sum',
        'timestamp': ['min', 'max']
    })

    agg_df.columns = ['account_id', 'total_amount_usd', 'avg_amount_usd', 'std_amount_usd',
                      'tx_count', 'tx_freq', 'tx_freq_std',
                      'asset_diversity', 'borrow_count', 'repay_count',
                      'first_seen', 'last_seen']

    agg_df.fillna(0, inplace=True)
    agg_df['behavior_consistency'] = agg_df['tx_freq'] * agg_df['tx_count']
    agg_df['borrow_repay_ratio'] = agg_df['borrow_count'] / (agg_df['repay_count'] + 1)
    agg_df['active_days'] = (agg_df['last_seen'] - agg_df['first_seen']).dt.days + 1

    return agg_df

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        print("⚠ No data to normalize. Returning empty DataFrame.")
        return df
    id_col = 'account_id'
    numeric = df.drop(columns=[id_col, 'first_seen', 'last_seen'], errors='ignore')
    numeric = numeric.select_dtypes(include=[np.number])
    if numeric.empty:
        print("⚠ No numeric data to normalize. Returning original DataFrame.")
        return df
    scaler = MinMaxScaler()
    try:
        normalized_values = scaler.fit_transform(numeric)
        normalized = pd.DataFrame(normalized_values, columns=numeric.columns)
    except ValueError as e:
        print(f"⚠ Normalization failed: {e}")
        return df
    normalized[id_col] = df[id_col].values
    return normalized[[id_col] + list(numeric.columns)]

def cluster(data):
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    return kmeans.fit_predict(data)

def score(df, labels):
    df = df.copy()
    df['cluster'] = labels
    df['deposit_volume'] = df['total_amount_usd']
    df['risk_score'] = df['borrow_repay_ratio']
    major_cluster = df.groupby('cluster')['deposit_volume'].mean().idxmax()
    df['credit_score'] = 100 * (df['cluster'] == major_cluster).astype(int)
    df['credit_score'] += df['risk_score'] * 10
    df['credit_score'] += df['asset_diversity'] * 2
    df['credit_score'] += df['active_days'] * 0.1
    df['credit_score'] -= df['behavior_consistency'] * 0.01
    df['credit_score'] = df['credit_score'].clip(0, 100)
    return df

def analyze_wallets(df):
    top_wallets = df.sort_values(by='credit_score', ascending=False).head(5)
    low_wallets = df.sort_values(by='credit_score').head(5)

    print("\n🌟 High-Scoring Wallets (Top 5):")
    print(top_wallets[['account_id', 'credit_score', 'tx_count', 'asset_diversity', 'borrow_repay_ratio', 'active_days']])

    print("\n⚠️ Low-Scoring Wallets (Bottom 5):")
    print(low_wallets[['account_id', 'credit_score', 'tx_count', 'asset_diversity', 'borrow_repay_ratio', 'active_days']])

def main():
    data_dir = "data"
    json_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json")]
    raw_df = load_all_json(json_files)
    raw_df = preprocess(raw_df)
    wallet_features = engineer(raw_df)
    if wallet_features.empty:
        print("❌ No features generated. Exiting.")
        return
    norm_data = normalize(wallet_features)
    if norm_data.empty:
        print("❌ Normalization returned empty data. Exiting.")
        return
    features_for_clustering = norm_data.drop(columns=['account_id'], errors='ignore')
    labels = cluster(features_for_clustering)
    final_df = score(wallet_features, labels)

    # ✅ Output CSV
    final_df.sort_values(by='credit_score', ascending=False).head(1000).to_csv('top_1000_wallets.csv', index=False)
    print("\n✅ Saved top 1000 wallets to top_1000_wallets.csv")

    # ✅ Print Analysis
    analyze_wallets(final_df)

if __name__ == "__main__":
    main()
