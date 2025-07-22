import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split

# STEP 1: Generate synthetic dataset
X, y = make_classification(
    n_samples=20000,
    n_features=30,
    n_informative=12,             # More signal
    n_redundant=5,                # Some correlated noise
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=2,       # Simplify geometry a bit
    weights=[0.985, 0.015],       # Still very imbalanced
    flip_y=0.02,                  # Minimal label noise
    class_sep=1.0,                # Decent class separation
    scale=1.0,
    shuffle=True,
    random_state=42
)


df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
df['is_fraud'] = y

# STEP 2: EDA
print("Class distribution:\n", df['is_fraud'].value_counts(normalize=True))

# Correlation matrix
plt.figure(figsize=(14, 10))
corr = df.corr()
sns.heatmap(corr.iloc[:-1, :-1], cmap="coolwarm", center=0, cbar_kws={'label': 'Correlation'})
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Feature distribution by class
df_melt = df.copy()
df_melt['is_fraud'] = df_melt['is_fraud'].map({0: "Legit", 1: "Fraud"})
sns.pairplot(df_melt.sample(1000), hue="is_fraud", vars=[f"f_{i}" for i in range(5)])
plt.suptitle("Feature Pairplots by Class", y=1.02)
plt.show()

# STEP 3: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns='is_fraud'),
    df['is_fraud'],
    test_size=0.3,
    stratify=df['is_fraud'],
    random_state=42
)

# STEP 4: Train a model
clf = LGBMClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# STEP 5: Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC Score: {roc_auc:.4f}")

RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve")
plt.show()