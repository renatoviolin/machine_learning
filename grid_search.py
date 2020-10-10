# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
np.set_printoptions(suppress=True)  # evitar notacao cientifica


# %% Lendo o dataset
df = pd.read_csv('dataset/Social_Network_Ads.csv')
df.head()
len(df)
df.drop('User ID', axis=1, inplace=True)


# %% One Hot encoding para a coluna Gender
ohe = OneHotEncoder(drop='first')
one_hot = ohe.fit_transform(df.Gender.values.reshape(-1, 1)).toarray()
df.insert(1, 'gender', one_hot)
df.drop('Gender', axis=1, inplace=True)


# %% Feature Scaling das colunas Age e EstimatedSalary
sc = StandardScaler()
df.iloc[:, 1:3] = sc.fit_transform(df.iloc[:, 1:3])

# trazendo de volta os valores na escala original
sc.inverse_transform(df.iloc[:, 1:3])


# %% Treino e Validacao
X = df.iloc[:, 0:3].values
y = df.iloc[:, 3].values
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=0)
print(f'Dados de treino: {x_train.shape[0]} samples')
print(f'Dados de validação.: {x_val.shape[0]} samples')


# %% Definindo o modelo SVC e treinando primeiro modelo para usar como baseline
model = SVC(kernel='linear')
model.fit(X=x_train, y=y_train)

y_pred_train = model.predict(x_train)
y_pred_val = model.predict(x_val)

acc_train = np.sum(y_pred_train == y_train) / len(y_train)
acc_val = np.sum(y_pred_val == y_val) / len(y_val)
print(f'Train acc: {acc_train:.4f}')
print(f'Val acc..: {acc_val:.4f}')


# %% Aplicando o grid search para buscar os melhores hyper-parametros para o modelo
# os hyper-parametros são específicos para cada modelo
params = [
    {'kernel': ['linear'], 'gamma': [0.1, 0.5, 0.9]},
    {'kernel': ['rbf'], 'gamma': [0.1, 0.5, 0.9]},
    {'kernel': ['sigmoid'], 'gamma': [0.1, 0.5, 0.9]},
    {'kernel': ['poly'], 'degree': [6, 12, 21]}
]
grid_search = GridSearchCV(estimator=model,
                           param_grid=params,
                           scoring='accuracy',
                           cv=5,
                           n_jobs=-1)


results = grid_search.fit(x_train, y_train)
best_score = grid_search.best_score_
best_parameters = grid_search.best_params_
cv_res = grid_search.cv_results_
print('\nBEST: %.3f - %s \n' % (best_score, best_parameters))


# %% Exibindo todos os scores
for score, params in zip(cv_res['mean_test_score'], cv_res['params']):
    print('%.3f - %s' % (score, params))


# %% Repete o modelo agora usando melhores os hyper-parametros
best_model = SVC(gamma=0.5, kernel='rbf')
best_model.fit(X=x_train, y=y_train)

y_pred_train = best_model.predict(x_train)
y_pred_val = best_model.predict(x_val)

acc_train = np.sum(y_pred_train == y_train) / len(y_train)
acc_val = np.sum(y_pred_val == y_val) / len(y_val)
print('===== RESULTADOS COM OS MELHORES HYPER-PARAMETROS ======')
print(f'Train acc: {acc_train:.4f}')
print(f'Val acc..: {acc_val:.4f}')

# %%
