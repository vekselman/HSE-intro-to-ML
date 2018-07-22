# coding=utf-8
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime
from sklearn.linear_model import LogisticRegression

# Part 1 gradient boosting
print 'Gradient boosting'

train_X = pd.read_csv('features.csv')
test_data = pd.read_csv('features_test.csv')

train_y = train_X['radiant_win']
train_X = train_X.drop(
    ['tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire', 'duration',
     'radiant_win'], 1)

print 'attributes with gaps \n{}'.format(train_X.count()[lambda x: x < len(train_X)])

train_X = train_X.fillna(0).as_matrix()

cv = KFold(len(train_y), n_folds=5, shuffle=True, random_state=241)
for estimators in range(10, 31, 10):
    clf = GradientBoostingClassifier(n_estimators=estimators, random_state=241)

    start_time = datetime.datetime.now()
    auc_score = []

    for traincv, testcv in cv:
        clf.fit(train_X[traincv], train_y[traincv])
        pred = clf.predict_proba(train_X[testcv])[:, 1]
        auc_score.append(metrics.roc_auc_score(train_y[testcv], pred))

    elapsed_time = datetime.datetime.now() - start_time
    print 'estimators: {0} , auc score: {1:.2f}, time elapsed: {2}'.format(estimators, np.mean(auc_score), elapsed_time)

# Part 2 logistic regression
print "Logistic regression"

heroes_columns = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero',
                  'd5_hero']


def make_hero_features(data, n):
    features = np.zeros((data.shape[0], n))
    for i, match_id in enumerate(data.index):
        for p in xrange(5):
            features[i, data.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
            features[i, data.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
    return features


def drop_category_features(data):
    return data.drop(
        ['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero',
         'd4_hero',
         'd5_hero'], 1)


def cross_val_score(train_X, train_y, cv):
    reg = np.power(10.0, np.arange(-5, 5))
    for c in reg:
        clf = LogisticRegression(C=c, random_state=241)
        start_time = datetime.datetime.now()
        auc_score = []
        for traincv, testcv in cv:
            clf.fit(train_X[traincv], train_y[traincv])
            pred = clf.predict_proba(train_X[testcv])[:, 1]
            auc_score.append(metrics.roc_auc_score(train_y[testcv], pred))
        elapsed_time = datetime.datetime.now() - start_time
        print 'C: {0} , auc score: {1:.2f}, time elapsed: {2}'.format(c, np.mean(auc_score), elapsed_time)


scaler = StandardScaler()

train_X = pd.read_csv('features.csv').fillna(0)
test_data = pd.read_csv('features_test.csv').fillna(0)

train_y = train_X['radiant_win']
train_X = train_X.drop(
    ['tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire', 'duration',
     'radiant_win'], 1)

train_X_t = scaler.fit_transform(train_X)
cv = KFold(len(train_y), n_folds=5, shuffle=True, random_state=241)

print "for original data"
cross_val_score(train_X_t, train_y, cv)

train_X_cut = drop_category_features(train_X)
train_X_t = scaler.fit_transform(train_X_cut)

print "after drop category features"
cross_val_score(train_X_t, train_y, cv)

heroes = train_X.loc[:, heroes_columns]

unique_heroes = []
for i in heroes_columns:
    unique_heroes.extend(heroes[i].unique())

unique_heroes = set(unique_heroes)
N = max(unique_heroes)
print 'unique heroes: {}'.format(N)
X_pick = make_hero_features(train_X, N)

train_X_stack = scaler.fit_transform(np.hstack((train_X_cut, X_pick)))
print "after translate category features to numeric"
cross_val_score(train_X_stack, train_y, cv)

print "Logistic regression for best of the best model"
clf = LogisticRegression(C=0.01, random_state=241)

clf.fit(train_X_stack, train_y)

X_pick = make_hero_features(test_data, N)
test_data = drop_category_features(test_data)

test_X_stack = scaler.transform(np.hstack((test_data, X_pick)))
predicted = clf.predict_proba(test_X_stack)[:, 1]
print 'probability max {}, min {}'.format(predicted.max(), predicted.min())