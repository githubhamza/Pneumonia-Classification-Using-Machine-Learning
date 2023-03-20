import numpy as np
import torch
from skimage import transform
import os
from imageio import imread


def img_resizer(img, target_shape):
    resized = transform.resize(
        np.float32(img) / 255, target_shape,
        order=2,
        anti_aliasing=True,
        clip=True
    )
    return np.float32(resized)


train_path = (
    '/content/sample_data/dataset/train/PNEUMONIA',
    '/content/sample_data/dataset/train/NORMAL'
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('/content/drive/MyDrive/ML_Project/weights/cnn.pth').feature_extractor.to(device) #enter path for saved weights from train_cnn

with torch.no_grad():
    model.eval()
    dataset = []
    for f in os.listdir(train_path[0]):
        print(f)
        img = imread(os.path.join(train_path[0], f))
        resized = img_resizer(img, (256, 256, 3))
        resized = resized.reshape((1, 3, 256, 256))
        inp = torch.from_numpy(resized).to(device)
        features = model.forward(inp).cpu().numpy()
        features = np.concatenate((features[0], np.array([0])), axis=0)
        dataset.append(features)

    print(len(dataset))
    for f in os.listdir(train_path[1]):
        print(f)
        img = imread(os.path.join(train_path[1], f))
        resized = img_resizer(img, (256, 256, 3)).reshape((3, 256, 256))
        resized = resized.reshape((1, 3, 256, 256))
        inp = torch.from_numpy(resized).to(device)
        features = model.forward(inp).cpu().numpy()
        features = np.concatenate((features[0], np.array([1])), axis=0)
        dataset.append(features)
    print(len(dataset))

dataset = np.asarray(dataset)
np.random.shuffle(dataset)

np.save('/content/drive/MyDrive/ML_Project/dataset/deep_f_train.npy', dataset) #enter path for saving features from ext_deepfeatures
