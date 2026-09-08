import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                              confusion_matrix, recall_score, precision_score, f1_score)

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
TEAL = '#0f766e'
DARK = '#134e4a'

df = pd.read_csv('cleaned_data.csv')

# --- Build feature matrix ---
drop_for_model = ['patient_nbr', 'readmitted', 'readmit_30', 'age',
                   'inpatient_bucket' if 'inpatient_bucket' in df.columns else 'age']
drop_for_model = list(set([c for c in drop_for_model if c in df.columns]))

y = df['readmit_30']
X = df.drop(columns=drop_for_model + ['readmit_30'])

cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(exclude='object').columns.tolist()
print(f"Categorical cols: {len(cat_cols)}, Numeric cols: {len(num_cols)}")

# Label-encode categoricals (tree models handle this fine; keeps dimensionality low vs one-hot)
X_enc = X.copy()
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    X_enc[c] = le.fit_transform(X_enc[c].astype(str))
    encoders[c] = le

X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train positive rate: {y_train.mean():.3f}, Test positive rate: {y_test.mean():.3f}")

results = {}

def evaluate(name, model, X_te, y_te, proba):
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_te, proba)
    rec = recall_score(y_te, pred)
    prec = precision_score(y_te, pred)
    f1 = f1_score(y_te, pred)
    results[name] = dict(auc=auc, recall=rec, precision=prec, f1=f1,
                          proba=proba, pred=pred)
    print(f"\n== {name} ==")
    print(f"ROC-AUC: {auc:.3f} | Recall(<30d): {rec:.3f} | Precision(<30d): {prec:.3f} | F1: {f1:.3f}")
    print(classification_report(y_te, pred, target_names=['No <30d', '<30d readmit']))

# --- Model 1: Random Forest (with class_weight to address imbalance) ---
rf = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=20,
    class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]
evaluate('Random Forest', rf, X_test, y_test, rf_proba)

# --- Model 2: XGBoost (with scale_pos_weight to address imbalance) ---
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    scale_pos_weight=scale_pos_weight, eval_metric='logloss',
    random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)
xgb_proba = xgb.predict_proba(X_test)[:, 1]
evaluate('XGBoost', xgb, X_test, y_test, xgb_proba)

# --- Baseline: Logistic Regression for reference ---
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_sc, y_train)
lr_proba = lr.predict_proba(X_test_sc)[:, 1]
evaluate('Logistic Regression (baseline)', lr, X_test, y_test, lr_proba)

# --- Comparison table ---
comp = pd.DataFrame({k: {m: round(v, 3) for m, v in d.items() if m != 'proba' and m != 'pred'}
                      for k, d in results.items()}).T
comp.to_csv('model_comparison.csv')
print("\n=== MODEL COMPARISON ===")
print(comp)

# --- ROC curve plot ---
fig, ax = plt.subplots(figsize=(6.5, 5.5))
colors = {'Random Forest': TEAL, 'XGBoost': DARK, 'Logistic Regression (baseline)': '#94a3b8'}
for name, d in results.items():
    fpr, tpr, _ = roc_curve(y_test, d['proba'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={d['auc']:.3f})", color=colors.get(name), linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve Comparison: RF vs XGBoost vs Logistic Regression')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('fig7_roc_comparison.png')
plt.close()

# --- Confusion matrices ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, name in zip(axes, ['Random Forest', 'XGBoost']):
    cm = confusion_matrix(y_test, results[name]['pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='BuGn', ax=ax, cbar=False,
                xticklabels=['No <30d', '<30d'], yticklabels=['No <30d', '<30d'])
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('fig8_confusion_matrices.png')
plt.close()

# --- Feature importance (RF and XGBoost) ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
rf_imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(12)
xgb_imp = pd.Series(xgb.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(12)
rf_imp.sort_values().plot(kind='barh', ax=axes[0], color=TEAL)
axes[0].set_title('Random Forest - Top 12 Feature Importances')
xgb_imp.sort_values().plot(kind='barh', ax=axes[1], color=DARK)
axes[1].set_title('XGBoost - Top 12 Feature Importances')
plt.tight_layout()
plt.savefig('fig9_feature_importance.png')
plt.close()

print("\nTop RF features:\n", rf_imp.round(3))
print("\nTop XGB features:\n", xgb_imp.round(3))
print("\nAll model figures saved.")