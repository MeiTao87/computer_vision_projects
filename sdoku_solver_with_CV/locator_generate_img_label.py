'''
This code is used to generate training images and labels 
for a NN to locate the sudoku puzzle in an given image
'''

import cv2
import albumentations as A
import os
import matplotlib.pyplot as plt
import json
import numpy as np

def generate_labels(H=1599, W=899):
    full_path = os.path.realpath(__file__)
    img_dir = os.path.dirname(full_path) + '/data/' + 'img/'
    label_dir = os.path.dirname(full_path) + '/data/' + 'label/'
    json_dir = os.path.dirname(full_path) + '/data/' + 'json/'
    
    for img_name in sorted(os.listdir(img_dir)):
        label_array = np.zeros((1, 10), dtype=np.float32)
        img_name = img_name[:-5]
        label_json_name = img_name + '.json'
        label_array_name = img_name + '.npy'
        label_json_path = json_dir + label_json_name
        label_array_path = label_dir + label_array_name
        
        # if there is a json file with same name, first two digits will be 1, 0
        # otherwise be 0, 1
        if os.path.exists(label_json_path):
            label_array[0][0] = 1
            # read json
            with open(label_json_path, 'r') as label_json:
                coordinate = label_json.read()
            coordinate_dict = json.loads(coordinate)
            for i in range(len(coordinate_dict['shapes'])):
                coor = coordinate_dict['shapes'][i]
                x, y = int(coor['points'][0][0]), int(coor['points'][0][1])
                x_width_ratio = x / W
                y_height_ratio = y /H
                label_array[0][2+2*i] = x_width_ratio
                label_array[0][3+2*i] = y_height_ratio 
            # print('exist, label is: ', label_array)
        else:
            label_array[0][1] = 1
        print(f'Saving {label_array_name} to the folder!')    
        np.save(label_array_path, label_array)

def generate_training_for_locator(num_of_training, h=160*3, w=270):
    full_path = os.path.realpath(__file__)
    img_dir = os.path.dirname(full_path) + '/data/' + 'img/'
    label_dir = os.path.dirname(full_path) + '/data/' + 'label/'   
    training_data = np.empty((num_of_training, h, w), dtype=np.uint8)
    training_label = np.empty((num_of_training, 1, 10), dtype=np.float32)
    
    transform = A.Compose([
        A.GaussNoise(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.5),
        ])

    if len(os.listdir(img_dir)) != len(os.listdir(label_dir)):
        print('Number of images and labels is not the same!')
    elif len(os.listdir(img_dir)) == len(os.listdir(label_dir)):
        for i in range(num_of_training):
            print(f'Generating the {i} img and label!')
            # randomly choose one image from the folder
            index = np.random.randint(0, len(os.listdir(img_dir)))
            chosen_img = sorted(os.listdir(img_dir))[index]
            chosen_img_label = sorted(os.listdir(label_dir))[index]
            
            chosen_img_path = os.path.join(img_dir, chosen_img)
            chosen_img_label_path = os.path.join(label_dir, chosen_img_label)
            
            chosen_img = cv2.imread(chosen_img_path, 0)
            # add augmentation to the chosen_img here
            chosen_img = transform(image=chosen_img)['image']
            # resize the image to save space
            chosen_img = cv2.resize(chosen_img, (w, h))
            chosen_label = np.load(chosen_img_label_path)
            training_data[i] = chosen_img
            training_label[i] = chosen_label
    return training_data, training_label

if __name__ == '__main__':
    
    generate_labels()
    training_data, training_label = generate_training_for_locator(5000)
    full_path = os.path.realpath(__file__)
    save_path = os.path.dirname(full_path) + '/data/'
    # np.savez_compressed("random.npz", img=img)
    np.savez_compressed(save_path+'training.npz', img=training_data)
    np.savez_compressed(save_path+'label.npz', label=training_label)
    
