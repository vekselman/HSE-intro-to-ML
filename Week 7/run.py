# scikit-learn==0.18

from __future__ import print_function

from datetime import datetime

from numpy import hstack, logspace, zeros
from pandas import DataFrame, read_csv
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import scale

GRADIENT_BOOSTING_JOBS = 4
LOGISTIC_REGRESSION_JOBS = 2
FINAL_ALGORITHM_JOBS = 4

# Part 1. Gradient boosting

# 1
DROP_COLUMNS = [
    'duration',
    'tower_status_radiant',
    'tower_status_dire',
    'barracks_status_radiant',
    'barracks_status_dire'
]


def read_data(path):
    features = read_csv(path, index_col='match_id')
    features.drop('start_time', axis=1)
    return features

features = read_data('./features.csv')
features = features.drop(DROP_COLUMNS, axis=1)

print('\nTrain data. rows: {0}, features: {1} '.format(
    features.shape[0], features.shape[1]))


# 2
print('\nFeatures with gaps:')
max_count = len(features)
for key, value in features.count().iteritems():
    if value >= max_count:
        continue
    print("Name: {0}, count: {1}".format(key, value))

# 3
features.fillna(0, inplace=True)

# 4
target = 'radiant_win'
print('\nTarget value {0}'.format(target))

y = features.pop('radiant_win').values
X = features

# # 5
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for n_estimators in [10, 20, 30, 40]:
    params = {
        'n_estimators': n_estimators,
        'max_depth': 5,
        'random_state': 241
    }
    start_time = datetime.now()

    model = GradientBoostingClassifier(**params)
    probas = cross_val_predict(model, X, y, cv=kf,
                               n_jobs=GRADIENT_BOOSTING_JOBS)
    score = roc_auc_score(y, probas)

    print('Trees: {0}, time:{1}, ROC-AUC: {2:.2f}'.format(
        n_estimators, datetime.now() - start_time, score))


# Part 2. Logistic regression

# 1

def logistic_regression_model(X, y, kf, n_jobs=4):
    b_score = -1
    b_c = 0
    b_time = 0
    for c in logspace(-3, 0, num=5):
        start_time = datetime.now()

        params = {
            'penalty': 'l2',
            'n_jobs': n_jobs,
            'C': c
        }
        model = LogisticRegression(**params)
        probas = cross_val_predict(model, X_scaled, y, cv=kf, n_jobs=n_jobs)
        score = roc_auc_score(y, probas)

        work_time = datetime.now() - start_time

        print('C={0:.5f}, time:{1}, ROC-AUC: {2:.4f}'.format(
            c, work_time, score))

        if score > b_score:
            b_score = score
            b_c = c
            b_time = work_time

    return b_score, b_c, b_time

print('\n')
X_scaled = scale(X, copy=False)
score, c, time = logistic_regression_model(X_scaled, y, kf,
                                           n_jobs=LOGISTIC_REGRESSION_JOBS)
print(
    '\nAll features. C={0:.5f}, time:{1}, ROC-AUC:{2:.4f}'.format(
        c, time, score)
)


# 2
NON_HERO_FEATURES = [
    'lobby_type'
]

HERO_FEATURES = [
    'r1_hero',
    'r2_hero',
    'r3_hero',
    'r4_hero',
    'r5_hero',
    'd1_hero',
    'd2_hero',
    'd3_hero',
    'd4_hero',
    'd5_hero'
]

non_hero_features = features.drop(NON_HERO_FEATURES, axis=1)
dropped_features = features.drop(NON_HERO_FEATURES + HERO_FEATURES, axis=1)
X_scaled = scale(dropped_features, copy=False)
score, c, time = logistic_regression_model(X_scaled, y, kf,
                                           n_jobs=LOGISTIC_REGRESSION_JOBS)
print(
    '\nWithout categorial features. C={0:.5f}, time:{1}, ROC-AUC: {2:.4f}'.format(
        c, time, score)
)


# 3
def get_unique_heroes(features):
    unique_heroes = set()

    for name in HERO_FEATURES:
        unique_heroes.update(features[name].value_counts().index.values)

    return list(unique_heroes)

unique_heroes = get_unique_heroes(non_hero_features)
print('\n{0} unique heroes'.format(len(unique_heroes)))


# 4
def get_new_hero_features(features, unique_heroes):
    X_new_hero_features = zeros((features.shape[0], len(unique_heroes)))

    for i, match_id in enumerate(features.index):
        for p in xrange(5):
            r_hero = features.ix[match_id, 'r%d_hero' % (p + 1)]
            X_new_hero_features[i, unique_heroes.index(r_hero)] = 1

            d_hero = features.ix[match_id, 'd%d_hero' % (p + 1)]
            X_new_hero_features[i, unique_heroes.index(d_hero)] = -1

    return X_new_hero_features

X_new_hero_features = get_new_hero_features(
    non_hero_features, unique_heroes)


# 5
X_scaled = hstack((X_scaled, X_new_hero_features))

score, c, time = logistic_regression_model(X_scaled, y, kf,
                                           n_jobs=LOGISTIC_REGRESSION_JOBS)
print(
    '\nNew features. C={0:.5f}, time:{1}, ROC-AUC: {2:.4f}'.format(
        c, time, score)
)


# Part 3. Best algorithm

features = read_data('./features_test.csv')

print('\nTest data. {0} rows, {1} features'.format(
    features.shape[0], features.shape[1]))

features.fillna(0, inplace=True)

unique_heroes_test = get_unique_heroes(features)
X_new_hero_features_test = get_new_hero_features(features, unique_heroes_test)
truncated_features_test = features.drop(
    NON_HERO_FEATURES + HERO_FEATURES, axis=1)
X_test_scaled = hstack(
    (scale(truncated_features_test), X_new_hero_features_test))

best_clf = LogisticRegression(penalty='l2', C=c, n_jobs=FINAL_ALGORITHM_JOBS)
best_clf.fit(X_scaled, y)

Y_pred = best_clf.predict(X_test_scaled)

df = DataFrame(index=truncated_features_test.index, columns=[target])
df[target] = best_clf.predict_proba(X_test_scaled)
df.to_csv('predictions.csv')
