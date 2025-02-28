import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Ensure stratify works by checking class distribution
unique, counts = np.unique(labels, return_counts=True)
if min(counts) < 2:
    print("Warning: At least one class has less than 2 samples. Stratified splitting may fail.")
    stratify_param = None
else:
    stratify_param = labels

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=stratify_param)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open("model.p", "wb")
pickle.dump({'model': model}, f)
f.close()