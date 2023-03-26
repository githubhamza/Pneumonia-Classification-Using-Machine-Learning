from sklearn.svm import LinearSVC
import numpy as np
import joblib

dataset_path = '/dataset/deep_f_train.npy' #enter path for extracted features from ext_deepfeatures
model_path = '/weights/svc.sav' #enter path for saving weights of training_svm

dataset = np.load(dataset_path)

X, y = dataset[:, :-1], np.uint8(dataset[:, -1])

classifier = LinearSVC()

classifier.fit(X, y)

joblib.dump(classifier, model_path)

y_pred = classifier.predict(X)
y_pred = np.where(y_pred<0.5, 0, 1)

correct = np.where(y == y_pred, 1, 0)

print(correct.sum()/correct.shape[0])
