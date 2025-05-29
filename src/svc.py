from preprocessing import *
from sklearn import svm

# model -3
svc_clf = svm.SVC(random_state = 7)
svc_clf.fit(x_train, y_train)
print(svc_clf.score(x_test, y_test)) # Score -> 1.00

from sklearn.metrics import classification_report
y_preds = svc_clf.predict(x_test)
print(classification_report(y_test, y_preds))

# dev set
devx = dev.drop('target', axis = 1)
devy = dev['target']

# to test overfitting
dev_preds = svc_clf.predict(devx)
print(svc_clf.score(devx, devy))
print(classification_report(devy, dev_preds))