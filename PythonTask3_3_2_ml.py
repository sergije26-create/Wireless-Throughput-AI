# ES96T Task 3.3.2 - ML model for throughput prediction
# trains polynomial regression on Bianchi-generated dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('dataset_task3.csv')
print(f"Column names: {list(df.columns)}")
print(df.head())

# input features and output label
# n_stations: number of contending stations (5 to 50)
# m: maximum backoff stage (3 or 5)
# CWmin: minimum contention window (32 or 128)
# data_rate_Mbps: PHY data rate (6,12,24,54 Mbps)
# payload_bytes: packet payload size (1000, 2000, 4000 bytes)
# throughput: effective throughput from Bianchi model (output label)
feature_cols = ['n_stations', 'm', 'CWmin', 'data_rate_Mbps', 'payload_bytes']
target_col   = 'throughput'

X = df[feature_cols].values
y = df[target_col].values

# 80/20 train/test split, fixed seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# polynomial regression pipeline (degree 2)
# PolynomialFeatures expands 5 inputs into quadratic and interaction terms
# StandardScaler normalises these since polynomial terms span very different ranges
# LinearRegression fits coefficients via ordinary least squares
degree = 2
model = Pipeline([
    ('poly',   PolynomialFeatures(degree=degree, include_bias=True)),
    ('scaler', StandardScaler()),
    ('lr',     LinearRegression())
])

model.fit(X_train, y_train)

# print top 10 coefficients by magnitude
feature_names = model.named_steps['poly'].get_feature_names_out(feature_cols)
coefficients = model.named_steps['lr'].coef_
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
print(coef_df.head(10))

# evaluate on train and test sets
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test  = mean_squared_error(y_test,  y_pred_test)
r2_train  = r2_score(y_train, y_pred_train)
r2_test   = r2_score(y_test,  y_pred_test)

print(f"Train -> MSE: {mse_train:.6f}, R2: {r2_train:.4f}")
print(f"Test  -> MSE: {mse_test:.6f},  R2: {r2_test:.4f}")

# plot 1: predicted vs actual and residuals
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_test, y_pred_test, alpha=0.6, color='steelblue', edgecolors='k', linewidths=0.4, s=40)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=1.5, label='Perfect prediction')
axes[0].set_xlabel('Actual Throughput', fontsize=12)
axes[0].set_ylabel('Predicted Throughput', fontsize=12)
axes[0].set_title(f'Predicted vs Actual (Test Set)\nR² = {r2_test:.4f}', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

residuals = y_test - y_pred_test
axes[1].scatter(y_pred_test, residuals, alpha=0.6, color='coral', edgecolors='k', linewidths=0.4, s=40)
axes[1].axhline(0, color='black', linewidth=1.5, linestyle='--')
axes[1].set_xlabel('Predicted Throughput', fontsize=12)
axes[1].set_ylabel('Residual (Actual - Predicted)', fontsize=12)
axes[1].set_title('Residual Plot', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Graph_task3_ml_results.png', dpi=150, bbox_inches='tight')
plt.show(block=False)

# plot 2: model predictions vs Bianchi data for Table I combinations
# fixed at 54 Mbps and 4000 byte payload
fig, ax = plt.subplots(figsize=(9, 5))
n_vals = np.arange(5, 55, 5)

param_plot = [
    (3,  32,  'blue',  'm=3, CWmin=32'),
    (5,  32,  'red',   'm=5, CWmin=32'),
    (3,  128, 'green', 'm=3, CWmin=128'),
]

for m_val, cw, colour, label in param_plot:
    mask = ((df['m'] == m_val) &
            (df['CWmin'] == cw) &
            (df['data_rate_Mbps'] == 54) &
            (df['payload_bytes'] == 4000))

    actual_n = df[mask]['n_stations'].values
    actual_t = df[mask]['throughput'].values

    # build input array matching all 5 features for model prediction
    X_pred = np.column_stack([
        n_vals,
        np.full(len(n_vals), m_val),
        np.full(len(n_vals), cw),
        np.full(len(n_vals), 54),
        np.full(len(n_vals), 4000)
    ])
    pred_t = model.predict(X_pred)

    ax.plot(actual_n, actual_t, 'o', color=colour, label=f'Actual {label}', markersize=6)
    ax.plot(n_vals, pred_t, '--', color=colour, label=f'Predicted {label}', linewidth=1.5)

ax.set_xlabel('Number of Stations', fontsize=12)
ax.set_ylabel('Effective Throughput', fontsize=12)
ax.set_title('Model Predictions vs Bianchi Data\n(54 Mbps, Payload=4000 bytes)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Graph_task3_ml_predictions.png', dpi=150, bbox_inches='tight')
plt.show()