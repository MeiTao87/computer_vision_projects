import torch
import torch.nn as nn
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.in_channels = in_channels
        self.densenet = self._create_densenet()
        self.yolo_layer = self._create_yolo_layer(**kwargs)
    
    def forward(self, x):
        x = self.densenet(x)
        x = self.yolo_layer(x)
        return x

    def _create_densenet(self):
        model = torchvision.models.densenet161(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        return nn.Sequential(*list(model.children())[:-1])

    def _create_yolo_layer(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(2208*14*14, 128), # after BatchNorm2d: ?, 2208, 14, 14
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(128, S * S * (C + B * 5)),
        )


def main():
    yolo = Yolov1(in_channels=3, split_size=7,num_boxes=2, num_classes=20).to(device)
    x = torch.randn(3,3,448,448).to(device)
    print(yolo(x).shape)

if __name__ == '__main__':
    main()