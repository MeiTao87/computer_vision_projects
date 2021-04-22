import torch
import os
import json
from PIL import Image
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


class VOCDataset(torch.utils.data.Dataset):
    @staticmethod
    def json_to_bbox(json_file_path):
        with open (json_file_path) as f:
            data = json.load(f)
            points = data['shapes'][0]['points']
            H, W = data['imageHeight'], data['imageWidth']
            cls_name = data['shapes'][0]['label']
            # cls_name = 'zwq'
            enc = OneHotEncoder(handle_unknown='ignore', )
            X = [['mt'], ['zwq']]
            enc.fit(X)
            cls_id = enc.transform([[cls_name]]).toarray()[0]

            x1, y1 = points[0]
            x2, y2 = points[1]
            w, h = x2-x1, y2-y1
            centerx, centery = x1 + 0.5*w, y1 + 0.5*h
            x, y, w, h = centerx/W, centery/H, w/W, h/H
        return cls_id, x, y, w, h
    
    def __init__(
        self, training_path, S=7, B=2, C=20, transform=None,
    ):
        self.training_path = training_path
        self.sub_dirs = os.listdir(training_path)

        self.img_path = []
        self.label_path = []
        for self.sub_dir in self.sub_dirs:
            for f in (os.listdir(os.path.join(self.training_path, self.sub_dir))):
                if f.endswith('jpg'): # iterate over all images
                    # img_path contains all images
                    self.img_path.append(os.path.join(self.training_path, self.sub_dir, f))
                    self.label_path.append(os.path.join(self.training_path, self.sub_dir, f[:-4] + '.json'))

        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        boxes = []
        boxes = VOCDataset.json_to_bbox(self.label_path[index])
        
        image = Image.open(self.img_path[index]).convert('RGB')
        # boxes = torch.tensor(boxes)

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        # for box in boxes:
        #     print(f'Here {box}')
        class_label, x, y, width, height = boxes
        class_label = class_label.astype(np.int32)

        # i,j represents the cell row and cell column
        i, j = int(self.S * y), int(self.S * x)
        x_cell, y_cell = self.S * x - j, self.S * y - i

        """
        Calculating the width and height of cell of bounding box,
        relative to the cell is done by the following, with
        width as the example:
        
        width_pixels = (width*self.image_width)
        cell_pixels = (self.image_width)
        
        Then to find the width relative to the cell is simply:
        width_pixels/cell_pixels, simplification leads to the
        formulas below.
        """
        width_cell, height_cell = (
            width * self.S,
            height * self.S,
        )

        # If no object already found for specific cell i,j
        # Note: This means we restrict to ONE object
        # per cell!
        if label_matrix[i, j, self.C] == 0:
            # Set that there exists an object
            label_matrix[i, j, self.C] = 1

            # Box coordinates
            box_coordinates = torch.tensor(
                [x_cell, y_cell, width_cell, height_cell]
            )

            label_matrix[i, j, self.C+1:self.C+5] = box_coordinates

            # Set one hot encoding for class_label
            label_matrix[i, j, class_label] = 1

        return image, label_matrix


