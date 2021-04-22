import torch
import torch.nn
from torch.utils.data import DataLoader
from dataset import VOCDataset
import torchvision.transforms as transforms

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

train_dataset = VOCDataset(training_path = '/home/mt/Desktop/For_github/computer_vision_projects/face_recognition/data', S=3, C=2, transform=transform)

train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        num_workers=2,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

for batch_idx, a in enumerate(train_loader):
    print(batch_idx)
    print(a[0].shape)
    print(a[1].shape)
    break