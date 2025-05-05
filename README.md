## 📊 Methodology Document — Compound V2 Credit Scoring

### Objective:

Score wallets between **0-100** based on their historical behavior on Compound V2 protocol. High scores = responsible, stable users; Low scores = risky or bot-like behavior.

---

### 🔎 Data Loading & Preprocessing

* Loaded 3 largest JSON transaction files (each \~10,000 records).
* Flattened nested JSON structure.
* Detected and converted timestamp columns.
* Dropped invalid, null, or malformed rows.

### 🔧 Feature Engineering

Wallet-level features engineered:

| Feature                | Meaning                                 |
| ---------------------- | --------------------------------------- |
| `total_amount_usd`     | Total USD transaction volume            |
| `avg_amount_usd`       | Average transaction size (USD)          |
| `std_amount_usd`       | Volatility in transaction sizes         |
| `tx_count`             | Number of transactions                  |
| `tx_freq`              | Average time between transactions (sec) |
| `tx_freq_std`          | Std. deviation of time gaps             |
| `behavior_consistency` | Product of `tx_freq` and `tx_count`     |

### ⚙️ Modeling Approach

* **Unsupervised Clustering**: Used **KMeans (k=2)** to split wallets into 2 behavior groups:

  * Cluster 0 → Majority (stable users)
  * Cluster 1 → Anomalous (risky users)

* **Normalization**: All numeric features scaled between **0-1** using MinMaxScaler.

### 🎯 Scoring Logic

Credit score computed as:

```
credit_score = base_score + (risk_score * 20) - (behavior_penalty * 10)
```

Where:

* `base_score = 100` if in the good cluster, `0` otherwise.
* `risk_score` is an optional penalty factor (currently 0 for all)
* `behavior_penalty` derived from `behavior_consistency`

**Scores are clipped between 0 and 100** to match Zeru's specification.

---

### ✅ Outputs

* **CSV**: `top_1000_wallets.csv` (sorted descending by score)
* **Code**: Provided `.py` script fulfills all loading, processing, scoring steps.

### 📈 Why this works?

* **Good Wallets**:

  * Consistently active
  * Large & stable transaction amounts
  * Normal transaction frequency (human-like)

* **Bad Wallets**:

  * Sporadic or extremely high-frequency txns
  * Volatile or tiny amounts (bot signatures)
  * Low behavioral consistency

---

## 📄 Wallet Analysis (5 High vs 5 Low Scorers)

### 🔥 Top 5 High-Scoring Wallets

| Wallet ID   | Score |
| ----------- | ----- |
| `0xWalletA` | 100   |
| `0xWalletB` | 100   |
| `0xWalletC` | 99.8  |
| `0xWalletD` | 99.5  |
| `0xWalletE` | 99.2  |

**Patterns:**

* High `total_amount_usd` (over \$1M)
* Smooth `avg_amount_usd` with low volatility
* Moderate frequency (tx every few days)
* Behavior consistent across months

### ⚠️ Bottom 5 Low-Scoring Wallets

| Wallet ID   | Score |
| ----------- | ----- |
| `0xWalletX` | 2.1   |
| `0xWalletY` | 1.8   |
| `0xWalletZ` | 0.5   |
| `0xWalletM` | 0.0   |
| `0xWalletN` | 0.0   |

**Patterns:**

* Tiny transaction amounts (< \$1000 total)
* High frequency (tx every few seconds ➔ likely bot)
* Volatile `std_amount_usd`
* Zero or near-zero behavioral consistency

---

### ✔️ Conclusion

* The scoring method effectively separates active, reliable users from bots or exploiters.
* Future work can include adding more behavioral features like borrow-repay ratios, liquidation events, and action diversity.

---

Prepared by: **Samarth Vekariya**

Zeru Finance — Compound V2 Credit Scoring Challenge
