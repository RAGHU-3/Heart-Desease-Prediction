from preprocessing import *
from sklearn.neighbors import KNeighborsClassifier

# Model-2
# Parameters taken from grid search best params.
knn_clf = KNeighborsClassifier(algorithm = 'auto',
                               leaf_size = 10,
                               n_neighbors = 2,
                               p = 2)
knn_clf.fit(x_train, y_train)
print(knn_clf.score(x_test, y_test))

from sklearn.metrics import classification_report
y_preds = knn_clf.predict(x_test)
print(classification_report(y_test, y_preds))

# dev set
devx = dev.drop('target', axis = 1)
devy = dev['target']

# to test overfitting
dev_preds = knn_clf.predict(devx)
print(knn_clf.score(devx, devy))
print(classification_report(devy, dev_preds))