import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from skimage import transform
import os
from imageio import imread
from sklearn.svm import LinearSVC
import numpy as np
import joblib


def img_resizer(img, target_shape):
    resized = transform.resize(
        np.float32(img) / 255, target_shape,
        order=2,
        anti_aliasing=True,
        clip=True
    )
    return np.float32(resized)


test_path = (
    'dataset/test/PNEUMONIA',
    'datase t/test/NORMAL'
)  # path to penumonia and normal test set

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('weights/cnn.pth',
                   map_location=device).feature_extractor.to(device)  # load weights from train_cnn

with torch.no_grad():
    model.eval()
    dataset = []
    for f in os.listdir(test_path[0]):
        print(f)
        img = imread(os.path.join(test_path[0], f))
        resized = img_resizer(img, (256, 256, 3))
        resized = resized.reshape((1, 3, 256, 256))
        inp = torch.from_numpy(resized).to(device)
        features = model.forward(inp).cpu().numpy()
        features = np.concatenate((features[0], np.array([0])), axis=0)
        dataset.append(features)

    print(len(dataset))
    for f in os.listdir(test_path[1]):
        print(f)
        img = imread(os.path.join(test_path[1], f))
        resized = img_resizer(img, (256, 256, 3)).reshape((3, 256, 256))
        resized = resized.reshape((1, 3, 256, 256))
        inp = torch.from_numpy(resized).to(device)
        features = model.forward(inp).cpu().numpy()
        features = np.concatenate((features[0], np.array([1])), axis=0)
        dataset.append(features)
    print(len(dataset))

dataset = np.asarray(dataset)
np.random.shuffle(dataset)

# enter path for saving features from test.py
np.save('dataset/deep_f_test.npy', dataset)

# enter path for loading features from ext_deepfeatures
dataset_path = 'dataset/deep_f_test.npy'
model_path = 'weights/svc.sav'  # enter path for loading weights from training_svm

dataset = np.load(dataset_path)

X, y = dataset[:, :-1], np.uint8(dataset[:, -1])

loaded_model = joblib.load(model_path)
y_pred = loaded_model.predict(X)
y_pred = np.where(y_pred < 0.5, 0, 1)

correct = np.where(y == y_pred, 1, 0)

print(correct.sum()/correct.shape[0])


cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%',
            vmin=0, cmap='mako', cbar=False)
plt.show()
