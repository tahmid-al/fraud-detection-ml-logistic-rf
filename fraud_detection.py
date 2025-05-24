import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─── Load dataset ──────────────────────────────────────────────────────────────
# Data source: https://www.kaggle.com/datasets/dalpozz/creditcard-fraud`
df = pd.read_csv("/Users/tahmidalkawsarchowdhury/Documents/fraud detection/data/Fraud.csv")

# ─── Preprocessing ─────────────────────────────────────────────────────────────
# Keep only the two types that can be fraudulent
df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])]

# Drop identifier columns
df = df.drop(columns=[col for col in ['nameOrig', 'nameDest'] if col in df.columns])

# Ensure target is integer
df['isFraud'] = df['isFraud'].astype(int)

# Split into X and y
X = df.drop('isFraud', axis=1)
# Remove any non-numeric columns (e.g., dates)
X = X.select_dtypes(include=['number'])
y = df['isFraud']

# ─── Feature Scaling ───────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─── Train/Test Split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ─── Model Training & CV ───────────────────────────────────────────────────────
lr = LogisticRegression(class_weight='balanced')
rf = RandomForestClassifier(n_estimators=10, max_depth=5, n_jobs=-1, class_weight='balanced')
skf = StratifiedKFold(n_splits=2)
lr_scores = cross_val_score(lr, X_train, y_train, cv=skf, scoring='recall')
rf_scores = cross_val_score(rf, X_train, y_train, cv=skf, scoring='recall')

print(f"LR CV recall scores: {lr_scores.mean():.2%}")
print(f"RF CV recall scores: {rf_scores.mean():.2%}")

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# ─── Evaluation ────────────────────────────────────────────────────────────────
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

print("\nLogistic Regression Report:")
print(classification_report(y_test, y_pred_lr))

print("\nRandom Forest Report:")
print(classification_report(y_test, y_pred_rf))

# ─── Confusion Matrix Plot (RF) ────────────────────────────────────────────────
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf),
            annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/rf_confusion_matrix.png")
print(X.dtypes)
