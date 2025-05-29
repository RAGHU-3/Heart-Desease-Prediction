from preprocessing import *
from sklearn.ensemble import RandomForestClassifier

# Model-4
rand_clf = RandomForestClassifier(random_state = 31)
rand_clf.fit(x_train, y_train)
print(rand_clf.score(x_test, y_test)) # Score -> 1.00

from sklearn.metrics import classification_report
y_preds = rand_clf.predict(x_test)
print(classification_report(y_test, y_preds))

# dev set
devx = dev.drop('target', axis = 1)
devy = dev['target']

# to test overfitting
dev_preds = rand_clf.predict(devx)
print(rand_clf.score(devx, devy))
print(classification_report(devy, dev_preds))