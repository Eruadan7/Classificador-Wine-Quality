import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from pickle import dump, load
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from pprint import pprint

# SUPPORT VECTOR MACHINE

# 1. CARREGAR OS DADOS
wine = pd.read_csv('wine.csv', sep=',')

# 2. SEPARAR ATRIBUTOS E CLASSE (antes da codificação)
dados_atributos = wine.drop('quality', axis=1)
dados_classe = wine['quality']

# 3. NORMALIZAÇÃO
# SVM é sensível à escala
scaler = StandardScaler()
dados_atributos_normalizados = scaler.fit_transform(dados_atributos)

# 4. BALANCEAMENTO COM SMOTE
resampler = SMOTE(random_state=42, k_neighbors=4)
atributos_b, classes_b = resampler.fit_resample(dados_atributos_normalizados, dados_classe)

# print("Distribuição após SMOTE:")
# print(pd.Series(classes_b).value_counts())

# 5. DEFINIR GRADE DE HIPERPARÂMETROS para SVM
# svm_grid = {
#     'C': [0.1, 1, 10, 100],                    # Regularização
#     'kernel': ['linear', 'rbf', 'poly'],       # Tipo de kernel
#     'gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Coeficiente do kernel
#     'degree': [2, 3, 4]                        # Apenas para kernel 'poly'
# }

# # 6. OTIMIZAÇÃO COM RANDOMIZEDSEARCHCV
# svm = SVC(random_state=42)

# svm_hyperparameters = RandomizedSearchCV(
#     estimator=svm,
#     param_distributions=svm_grid,
#     n_iter=10,
#     cv=3,
#     verbose=1,
#     n_jobs=-1,
#     random_state=42
# )

# svm_hyperparameters.fit(atributos_b, classes_b)

# print("\n=== MELHORES PARÂMETROS SVM ===")
# pprint(svm_hyperparameters.best_params_)

#salvar objeto randomized_search
#dump(svm_hyperparameters, open('svm_random_search.pkl', 'wb'))

# 7. INSTANCIAR MODELO OTIMIZADO
svm_hyperparameters = load(open('svm_random_search.pkl', 'rb'))
svm_otimizado = SVC(**svm_hyperparameters.best_params_, random_state=42)

# 8. VALIDAÇÃO CRUZADA FINAL (10 FOLDS)
scoring = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']

scores_cross = cross_validate(
    svm_otimizado,
    atributos_b, classes_b,
    scoring=scoring,
    n_jobs=-1,
    cv=10,
    verbose=0
)

print("\n=== RESULTADOS DA VALIDAÇÃO CRUZADA (cv=10) ===")
print(f"Acurácia média:  {scores_cross['test_accuracy'].mean():.4f} (+/- {scores_cross['test_accuracy'].std():.4f})")
# print(f"Precision média: {scores_cross['test_precision_weighted'].mean():.4f}")
# print(f"Recall médio:    {scores_cross['test_recall_weighted'].mean():.4f}")
print(f"F1-Score médio:  {scores_cross['test_f1_weighted'].mean():.4f}")

# 9. OBTER ACURÁCIA POR CLASSES

# Predições com validação cruzada

modelo_para_cv = SVC(**svm_hyperparameters.best_params_, random_state=42)
predicoes_cv = cross_val_predict(modelo_para_cv, atributos_b, classes_b, cv=10)

# Acurácia por classe = diagonal da matriz / total daquela classe 
matriz_cv = confusion_matrix(classes_b, predicoes_cv)
acuracia_por_classe_cv = matriz_cv.diagonal() / matriz_cv.sum(axis=1)

print("\n=== ACURÁCIA POR CLASSE (VALIDAÇÃO CRUZADA) ===")
qualidades = [3, 4, 5, 6, 7, 8, 9]
for i, qual in enumerate(qualidades):
    print(f"  Qualidade {qual}: {acuracia_por_classe_cv[i]:.4f}")

# 10. TREINAR MODELO FINAL
wine_svm = svm_otimizado.fit(atributos_b, classes_b)

# 11. SALVAR MODELO E O SCALER
dump(wine_svm, open('modelo_svm.pkl', 'wb'))
dump(scaler, open('wine_scaler.pkl', 'wb'))