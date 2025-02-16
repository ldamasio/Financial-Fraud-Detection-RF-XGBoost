import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import json

with open("financial_transactions.json", "r") as file:
    data = json.load(file)

df = pd.DataFrame(data)

# Exploração inicial dos dados
print(df.head())
print(df.info())
print(df.describe())

# Verificar valores ausentes
print("Valores ausentes por coluna:\n", df.isnull().sum())

# Remover valores ausentes se existirem
df.dropna(inplace=True)

# Definir variáveis independentes (X) e variável alvo (y)
X = df.drop(columns=['is_fraud'])  # Supondo que a coluna alvo seja 'is_fraud'
y = df['is_fraud']

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Treinar modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))
print("Random Forest AUC-ROC:", rf_auc)

# Treinar modelo XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
print("XGBoost Classification Report:\n", classification_report(y_test, xgb_pred))
print("XGBoost AUC-ROC:", xgb_auc)

# Visualizar matriz de confusão
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Random Forest Confusion Matrix")
sns.heatmap(confusion_matrix(y_test, xgb_pred), annot=True, fmt='d', cmap='Oranges', ax=axes[1])
axes[1].set_title("XGBoost Confusion Matrix")
plt.show()
