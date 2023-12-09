import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Use AdaBoostClassifier instead of MLPClassifier
model = AdaBoostClassifier(random_state=42)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predict)
print('Accuracy: {}% of samples were classified correctly!'.format(accuracy * 100))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_predict)
print('\nConfusion Matrix:')
print(conf_matrix)

# Save the model
with open('adaboost_model.p', 'wb') as f:
    pickle.dump({'model': model, 'confusion_matrix': conf_matrix}, f)
