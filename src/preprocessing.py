from input import *

# creating train, and dev set
train = df.loc[:900] # trainig set
dev = df.loc[901:] # development set to test overfitting

print(train.shape, dev.shape)
print(train.target.value_counts())
print(dev.target.value_counts())

# creating dependent and independent matrix of features
x = train.iloc[:, :-1]
y = train.iloc[:, -1]
print(x.shape, y.shape)

# create training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 31)