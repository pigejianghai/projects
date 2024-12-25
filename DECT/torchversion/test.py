import torch
import torchmetrics as tm
import os
from torchmetrics.classification import BinaryAUROC
from sklearn.model_selection import train_test_split



path = '/data1/jianghai/DECT/txt/binary/train.txt'
f = open(path, 'r')
id = []
label = []
for line in f:
    id.append(line.split()[0])
    label.append(line.split()[1])
f.close()
# print(label.count('0'), label.count('1'))
print(id)
print(label)

X_train, X_test, y_train, y_test = train_test_split(id, label, test_size=0.3, random_state=42)
# print(X_train)
print(y_train.count('0'), y_train.count('1'))
# print(X_test)
print(y_test.count('0'), y_test.count('1'))