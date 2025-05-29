from preprocessing import *
from sklearn.linear_model import LogisticRegression

# model-1
# Parameters taken from grid search best params.
log_clf = LogisticRegression(max_iter = 100,
                        C =  0.30888,
                        penalty= 'l2',
                        solver='liblinear',
                        random_state = 31)
log_clf.fit(x_train, y_train)
print(log_clf.score(x_test, y_test))

from sklearn.metrics import classification_report
y_preds = log_clf.predict(x_test)
print(classification_report(y_test, y_preds))

# dev set
devx = dev.drop('target', axis = 1)
devy = dev['target']

# to test overfitting
dev_preds = log_clf.predict(devx)
print(log_clf.score(devx, devy))
print(classification_report(devy, dev_preds))