import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix
from pickle import dump

# Gradient Boosting Classifier

# 1. CARREGAR OS DADOS
wine = pd.read_csv("wine.csv", sep=",")

# 2. SEPARAR ATRIBUTOS E CLASSE
dados_atributos = wine.drop('quality', axis=1)
dados_classe = wine['quality']

# 3. BALANCEAMENTO COM SMOTE
resampler = SMOTE(random_state=42, k_neighbors=4)
atributos_b, classes_b = resampler.fit_resample(dados_atributos, dados_classe)

# 4. GRADE DE HIPERPARÂMETROS
gb_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.7, 0.9, 1.0],
    'max_features': ['sqrt', 'log2']
}

# 5. OTIMIZAÇÃO COM RANDOMIZEDSEARCHCV
gb = GradientBoostingClassifier(random_state=42)

gb_hyperparameters = RandomizedSearchCV(
    estimator=gb,
    param_distributions=gb_grid,
    n_iter=10,
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

gb_hyperparameters.fit(atributos_b, classes_b)

# 6. INSTANCIAR MODELO OTIMIZADO
gb_otimizado = GradientBoostingClassifier(**gb_hyperparameters.best_params_, random_state=42)

# 7. VALIDAÇÃO CRUZADA FINAL (10 FOLDS)
scoring = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']

scores_cross = cross_validate(
    gb_otimizado,
    atributos_b, classes_b,
    scoring=scoring,
    cv=10,
    verbose=0
)

print("\n=== RESULTADOS DA VALIDAÇÃO CRUZADA (cv=10) ===")
print(f"Acurácia média:  {scores_cross['test_accuracy'].mean():.4f} (+/- {scores_cross['test_accuracy'].std():.4f})")
print(f"F1-Score médio:  {scores_cross['test_f1_weighted'].mean():.4f}")

# 8. ACURÁCIA POR CLASSE
modelo_para_cv = GradientBoostingClassifier(**gb_hyperparameters.best_params_, random_state=42)
predicoes_cv = cross_val_predict(modelo_para_cv, atributos_b, classes_b, cv=10)

matriz_cv = confusion_matrix(classes_b, predicoes_cv)
acuracia_por_classe_cv = matriz_cv.diagonal() / matriz_cv.sum(axis=1)

print("\n=== ACURÁCIA POR CLASSE (VALIDAÇÃO CRUZADA) ===")
qualidades = [3, 4, 5, 6, 7, 8, 9]
for i, qual in enumerate(qualidades):
    print(f"  Qualidade {qual}: {acuracia_por_classe_cv[i]:.4f}")

# 9. TREINAR MODELO FINAL
wine_gb = gb_otimizado.fit(atributos_b, classes_b)

# 10. SALVAR MODELO
dump(wine_gb, open('modelo_gradient_boosting.pkl', 'wb'))