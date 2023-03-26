from cnn import binary_classifier
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_root = '/dataset/train' #Enter training set path here
val_root = '/dataset/val' #Enter Validation set path here
img_channels = 3

image_transforms = {
    "train": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ]),
    "val": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
}

train_dataset = datasets.ImageFolder(
    root=train_root, 
    transform=image_transforms["train"]
)

val_dataset = datasets.ImageFolder(
    root=val_root, 
    transform=image_transforms["val"]
)

train_dataset_size = len(train_dataset)
train_dataset_indicies = list(range(train_dataset_size))

val_dataset_size = len(val_dataset)
val_dataset_indicies = list(range(val_dataset_size))

train_loader = DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=8
)

val_loader = DataLoader(
    dataset=val_dataset,
    shuffle=True,
    batch_size=1
)

model = binary_classifier(img_channels)
model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.008)

loss_stats = {
    'train': [],
    "val": []
}

for e in range(1, 21):
    train_epoch_loss = 0
    model.train(True)

    for idx, (X_train_batch, y_train_batch) in enumerate(train_loader):
        X_train_batch = X_train_batch.to(device)
        print(X_train_batch[0])
        y_train_batch = y_train_batch.type(torch.FloatTensor).to(device)
        optimizer.zero_grad()
        y_train_pred = model(X_train_batch).squeeze()
        train_loss = criterion(y_train_pred, y_train_batch)
        train_loss.backward()
        optimizer.step()
        train_epoch_loss += train_loss.item()
        if idx % 50 == 0: print('loss =', train_loss.item())
    print('EPOCH', e, 'COMPLETED')
    print('EPOCH LOSS =', train_epoch_loss)
    with torch.no_grad():
        model.eval()
        val_epoch_loss = 0
        val_epoch_acc = 0
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch = X_val_batch.to(device)
            y_val_batch = y_val_batch.type(torch.FloatTensor).to(device)
            y_val_pred = model(X_val_batch).squeeze()
            y_val_pred = torch.unsqueeze(y_val_pred, 0)
            val_loss = criterion(y_val_pred, y_val_batch)
            val_epoch_loss += val_loss.item()

    torch.save(model, '/weights/cnn.pth') #Enter path and filename for saving weights of train_cnn
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    
plt.plot(loss_stats['train'])
plt.plot(loss_stats['val'])
plt.show()
