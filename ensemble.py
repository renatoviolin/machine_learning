# %%
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
import numpy as np
np.random.seed(2)


# %%
df = pd.read_csv('dataset/imdb.csv')
df.sentiment = df.sentiment.apply(lambda x: 0 if x == 'negative' else 1)
df.insert(0, 'id', np.arange(len(df)))
df.head(10)


# %%
n_samples = 500  # utilize 25000 para considerar o dataset inteiro
g = df.groupby('sentiment')
df_small = g.apply(lambda x: x.sample(n_samples)).reset_index(drop=True)
df_small = df_small.sample(frac=1).reset_index(drop=True)
df_small.sentiment.value_counts()
df_small.loc[:, 'kfold'] = -1
y = df_small.sentiment.values


# %%
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds)
for f_idx, (t_idx, v_idx) in enumerate(skf.split(X=df_small, y=y)):
    df_small.loc[v_idx, 'kfold'] = f_idx

df_small[df_small.kfold == 0]


# %%
def run_lr(fold):
    tfv = TfidfVectorizer()

    df_train = df_small[df_small.kfold != fold].reset_index(drop=True)
    df_test = df_small[df_small.kfold == fold].reset_index(drop=True)

    tfv.fit(df_train.review.values)

    x_train = tfv.transform(df_train.review.values)
    y_train = df_train.sentiment.values

    x_test = tfv.transform(df_test.review.values)
    y_test = df_test.sentiment.values

    model = LogisticRegression()
    model.fit(X=x_train, y=y_train)
    y_pred = model.predict_proba(x_test)[:, 1]

    acc = np.sum((y_pred > 0.5) == y_test) / len(y_test)
    print(f'fold: {i} - Test acc: {acc}')

    df_test.loc[:, 'lr'] = y_pred
    return df_test[['id', 'sentiment', 'lr']]


_df = []
for i in range(n_folds):
    temp_df = run_lr(i)
    _df.append(temp_df)
df_lr = pd.concat(_df)


# %%
def run_svc(fold):
    tfv = TfidfVectorizer()

    df_train = df_small[df_small.kfold != fold].reset_index(drop=True)
    df_test = df_small[df_small.kfold == fold].reset_index(drop=True)

    tfv.fit(df_train.review.values)

    x_train = tfv.transform(df_train.review.values)
    y_train = df_train.sentiment.values

    x_test = tfv.transform(df_test.review.values)
    y_test = df_test.sentiment.values

    model = SVC(probability=True)
    model.fit(X=x_train, y=y_train)
    y_pred = model.predict_proba(x_test)[:, 1]

    acc = np.sum((y_pred > 0.5) == y_test) / len(y_test)
    print(f'fold: {i} - Test acc: {acc}')

    df_test.loc[:, 'svc'] = y_pred
    return df_test[['id', 'sentiment', 'svc']]


_df = []
for i in range(n_folds):
    temp_df = run_svc(i)
    _df.append(temp_df)
df_svc = pd.concat(_df)


# %%
def run_forest(fold):
    tfv = TfidfVectorizer()

    df_train = df_small[df_small.kfold != fold].reset_index(drop=True)
    df_test = df_small[df_small.kfold == fold].reset_index(drop=True)

    tfv.fit(df_train.review.values)

    x_train = tfv.transform(df_train.review.values)
    y_train = df_train.sentiment.values

    x_test = tfv.transform(df_test.review.values)
    y_test = df_test.sentiment.values

    model = RandomForestClassifier()
    model.fit(X=x_train, y=y_train)
    y_pred = model.predict_proba(x_test)[:, 1]

    acc = np.sum((y_pred > 0.5) == y_test) / len(y_test)
    print(f'fold: {i} - Test acc: {acc}')

    df_test.loc[:, 'forest'] = y_pred
    return df_test[['id', 'sentiment', 'forest']]


_df = []
for i in range(n_folds):
    temp_df = run_forest(i)
    _df.append(temp_df)
df_forest = pd.concat(_df)


# %%
# Merge dos dataset para analisar os resultados
_df_all = pd.merge(df_lr, df_svc, on='id', how='left')
df_all = pd.merge(_df_all, df_forest, on='id', how='left')[['id', 'sentiment', 'lr', 'svc', 'forest']]


# %%
# Exibe todos os resultados individuais
print('===== RESULTADOS INDIVIDUAIS =====')
cols = ['lr', 'svc', 'forest']
for c in cols:
    pred = df_all[c].values > 0.5
    target = df_all.sentiment.values
    acc = np.sum(pred == target) / len(target)
    print(f'Modelo: {c} \t {acc}')


# %%
# Media dos resultados
print('===== MEDIA ENTRE OS MODELOS =====')
pred_media = (df_all['lr'].values + df_all['svc'].values + df_all['forest'].values) / 3
target = df_all.sentiment.values
acc_media = np.sum((pred_media > 0.5) == target) / len(target)
print(f'Media: \t {acc_media}')


# %%
# Media ponderada dos resultados
print('===== MEDIA PONDERADA ENTRE OS MODELOS =====')
pred_media = (df_all['lr'].values * .3 + df_all['svc'].values * .5 + df_all['forest'].values * .2)
target = df_all.sentiment.values
acc_ponderada = np.sum((pred_media > 0.5) == target) / len(target)
print(f'Ponderada: \t {acc_ponderada}')


# %%
# Max entre os resultados
print('===== MAX PROBABILIDADE ENTRE OS MODELOS =====')
max_concat = np.concatenate([df_all['lr'].values.reshape(-1, 1),
                             df_all['svc'].values.reshape(-1, 1),
                             df_all['forest'].values.reshape(-1, 1)], axis=1)

pred_max = np.max(max_concat, axis=1)
target = df_all.sentiment.values
acc_max = np.sum((pred_max > 0.5) == target) / len(target)
print(f'Avg: \t {acc_max}')


# %%
# Pool entre os resultados
print('===== POOL ENTRE OS MODELOS =====')
target = df_all.sentiment.values

pred_lr = (df_all['lr'].values > 0.5) * 1
pred_svc = (df_all['svc'].values > 0.5) * 1
pred_forest = (df_all['forest'].values > 0.5) * 1

pool_concat = np.concatenate([pred_lr.reshape(-1, 1),
                              pred_svc.reshape(-1, 1),
                              pred_forest.reshape(-1, 1)], axis=1)

pool_pred = np.sum(pool_concat, axis=1) >= 2
acc_pool = np.sum(pool_pred == target) / len(target)
print(f'Avg: \t {acc_pool}')


# %%
