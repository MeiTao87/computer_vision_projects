import numpy as np
from utils import Solver, locate_sudoku
import matplotlib.pyplot as plt
import cv2
from model import DigitsClassifier
import tensorflow as tf
import os

# the following codes are needed for my laptop
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
'''sdoku = np.array(
   [[5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]])'''

# using CV to generate a 2D sdoku array


def main(from_camera=False):
    # digits_classifier = DigitsClassifier.build(width=28, height=28, depth=1, classes=10)
    full_path = os.path.realpath(__file__)
    save_dir = os.path.dirname(full_path) + '/weights/'
    digits_classifier = tf.keras.models.load_model(save_dir + 'digits_classifier.h5')
    model_img_size = (28, 28)
    sdoku = np.zeros((9, 9))
    
    if from_camera:
        cap = cv2.VideoCapture(1)
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # print('frame.shape', frame.shape) # 480, 640
            # print('type(frame)', type(frame)) # np.ndarray
            cv2.imshow('frame', frame)
            
            if locate_sudoku(frame) is not None: # if detect sudoku puzzle
                new_img = locate_sudoku(frame)
                H, W = new_img.shape
                h, w = H/9, W/9
                new_img = cv2.adaptiveThreshold(new_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 7)
                for col in range(9):
                    for row in range(9):
                        pt1 = (int(col*w+0.1*w), int(row*h+0.1*h))
                        pt2 = (int(col*w+0.9*w), int(row*h+0.9*h))
                        sub_img = new_img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
                        # using NN to tell which number it is, and put it in the input sdoku
                        # resized_sub_img = cv2.bitwise_not(sub_img)
                        # resized_sub_img = cv2.threshold
                        resized_sub_img = sub_img.astype("float32") / 255.0
                        resized_sub_img = cv2.resize(resized_sub_img, dsize=model_img_size)
                        resized_sub_img = np.expand_dims(resized_sub_img, axis=2)
                        resized_sub_img = np.expand_dims(resized_sub_img, axis=0)
                        predicted_num = digits_classifier.predict(resized_sub_img)
                        pre_array = np.array(predicted_num[0])
                        # print('argmax: ', np.argmax(pre_array))
                        # print( 'row', row, 'col', col)
                        # plt.imshow(sub_img, cmap='gray', vmin=0, vmax=1)
                        # plt.title('Predicted number is:' + str(np.argmax(pre_array)))
                        # plt.show()
                        sdoku[row, col] = np.argmax(pre_array)
                    print('unsolved puzzle: \n', sdoku)
            else:
                print('Did not find sudoku puzzle')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()    
    else:
        img = cv2.imread('/home/mt/Desktop/For_github/computer_vision_projects/sdoku_solver_with_CV/test_img/sudoku.jpg', 0)
        # fromImg = FromImg(img=img)
        # new_img = fromImg.convert()
        new_img = locate_sudoku(img)
        H, W = new_img.shape
        h, w = H/9, W/9
        # new_img = cv2.adaptiveThreshold(new_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 7)
        for col in range(9):
            for row in range(9):
                pt1 = (int(col*w+0.1*w), int(row*h+0.1*h))
                pt2 = (int(col*w+0.9*w), int(row*h+0.9*h))
                sub_img = new_img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
                # using NN to tell which number it is, and put it in the input sdoku
                # resized_sub_img = cv2.bitwise_not(sub_img)
                # resized_sub_img = cv2.threshold
                sub_img = sub_img.astype("float32") / 255.0
                resized_sub_img = cv2.resize(sub_img, dsize=model_img_size)
                resized_sub_img = np.expand_dims(resized_sub_img, axis=2)
                resized_sub_img = np.expand_dims(resized_sub_img, axis=0)
                predicted_num = digits_classifier.predict(resized_sub_img)
                pre_array = np.array(predicted_num[0])
                # print('argmax: ', np.argmax(pre_array))
                # print( 'row', row, 'col', col)
                # plt.imshow(sub_img, cmap='gray', vmin=0, vmax=1)
                # plt.title('Predicted number is:' + str(np.argmax(pre_array)))
                # plt.show()
                sdoku[row, col] = np.argmax(pre_array)
        print('unsolved puzzle: \n', sdoku)
    
    # solver = Solver(sdoku)
    # solver.solve()
    # print(sdoku)
if __name__ == '__main__':
    main(from_camera=False)