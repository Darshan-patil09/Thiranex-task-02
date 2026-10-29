from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score

ROOT = Path(__file__).resolve().parent
data = pd.read_csv(ROOT / "data" / "customer_churn.csv")
out = ROOT / "outputs"
out.mkdir(exist_ok=True)

X = data.drop(columns=["customer_id", "churn"])
y = data["churn"]
numeric = ["tenure_months", "monthly_charge", "support_tickets_3m"]
categorical = ["contract_type", "internet_service"]
preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
])
model = Pipeline([("preprocess", preprocess), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
metrics = {
    "test_rows": int(len(y_test)),
    "accuracy": round(float(accuracy_score(y_test, pred)), 4),
    "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
    "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
    "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 4),
    "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    "positive_class": "churn = 1",
    "data_note": "Synthetic demonstration dataset; metrics are for practice only."
}
(out / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
(out / "classification_report.txt").write_text(classification_report(y_test, pred, zero_division=0))
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix — Churn Model")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks([0,1], ["Stay", "Churn"])
plt.yticks([0,1], ["Stay", "Churn"])
for (i,j), v in __import__("numpy").ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")
plt.tight_layout()
plt.savefig(out / "confusion_matrix.png", dpi=150)
plt.close()
# Store a lightweight prediction example file, not a serialized model object.
sample = X_test.head(10).copy()
sample["actual_churn"] = y_test.loc[sample.index]
sample["predicted_churn"] = pred[:len(sample)]
sample["predicted_churn_probability"] = proba[:len(sample)].round(4)
sample.to_csv(out / "sample_predictions.csv", index=False)
print("Task 2 complete. Metrics:", json.dumps(metrics, indent=2))
