import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import os

def training_img_gennerate(training_img_num = 100):
    X = np.empty((training_img_num*81, 28, 28), dtype=int)
    y = np.empty((training_img_num*81, 1), dtype=int)
    img = cv2.imread('/home/mt/Desktop/For_github/computer_vision_projects/sdoku_solver_with_CV/test_img/sudoku_example.jpg', 0)
    img1 = img[73:547, 522:996]
    img2 = img[73:547, 26:500]
    # img = cv2.bitwise_not(img)
    img1 = cv2.resize(img1, (28*9, 28*9))
    img2 = cv2.resize(img2, (28*9, 28*9))
    
    H, W = img1.shape
    h, w = int(1/9*H), int(1/9*W)
    # image augmentation transformer
    transform = A.Compose([
        A.GaussNoise(p=0.5),
        # A.GridDistortion(p=0.3), # grid distortion
        A.ShiftScaleRotate(shift_limit=0.2, rotate_limit=0, p=0.5),
        A.RandomBrightnessContrast(p=0.5), # brightness contrast change
    ])

    label = np.array([[5, 4, 3, 9, 2, 1, 8, 7, 6], 
                      [2, 1, 9, 6, 8, 7, 5, 4, 3],
                      [8, 7, 6, 3, 5, 4, 2, 1, 9],
                      [9, 8, 7, 4, 6, 5, 3, 2, 1],
                      [3, 2, 1, 7, 9, 8, 6, 5, 4],
                      [6, 5, 4, 1, 3, 2, 9, 8, 7],
                      [7, 6, 5, 2, 4, 3, 1, 9, 8],
                      [4, 3, 2, 8, 1, 9, 7, 6, 5],
                      [1, 9, 8, 5, 7, 6, 4, 3, 2]], dtype=int)
    row_col_of_zero = [(0, 2), (0, 3), (0, 5), (0, 7), 
                        (1, 0), (1, 3), (1, 4), (1, 6), (1, 7),
                        (2, 0), (2, 1), (2, 2), (2, 4), (2, 5), (2, 8),
                        (3, 1), (3, 2), (3, 4), (3, 6), (3, 8),
                        (4, 0), (4, 1), (4, 3), (4, 4), (4, 5), (4, 7),
                        (5, 1), (5, 3), (5, 6), (5, 8),
                        (6, 0), (6, 2), (6, 3), (6, 4), (6, 5), (6, 8),
                        (7, 1), (7, 3), (7, 4), (7, 6), (7, 7), 
                        (8, 0), (8, 2), (8, 3), (8, 5), (8, 7)]
    for i in range(training_img_num):
        print('Finish: ', i+1, 'in ', training_img_num)
        for row in range(9):
            for col in range(10):
                if col < 9:
                    sub_img = img1[(row*w):(row*w+w), (col*h):(col*h+h)]    
                    sub_img = transform(image=sub_img)['image']
                    X[81*i + 9*row + col -1] = sub_img
                    y[81*i + 9*row + col -1] = label[row, col]
                else:
                    index = np.random.randint(0, len(row_col_of_zero))
                    row, col = row_col_of_zero[index]
                    sub_img = img2[row*w:row*w+w, col*h:col*h+h]  
                    X[81*i + 9*row + col -1] = sub_img
                    y[81*i + 9*row + col -1] = 0
                
                # print(label[row, col])
                # plt.imshow(sub_img, cmap='gray', vmin=0, vmax=255)
                # plt.show()
    return X, y
        
if __name__ == '__main__':
    full_path = os.path.realpath(__file__)
    save_dir = os.path.dirname(full_path) + '/training_img/'
    # check file existence
    if not os.path.exists(save_dir + 'X_train.npy'):
        try:
            os.mkdir(save_dir)
        except:
            pass
        X_train, y_train = training_img_gennerate(training_img_num=1000)
        X_test, y_test = training_img_gennerate(training_img_num=200)
        # save
        np.save(save_dir + 'X_train.npy', X_train) 
        np.save(save_dir + 'y_train.npy', y_train)
        np.save(save_dir + 'X_test.npy', X_test)
        np.save(save_dir + 'y_test.npy', y_test)
        print('Finished')
    else:
        print('Training data already existed!')
    a = np.load(save_dir + 'X_train.npy')
    b = np.load(save_dir + 'y_train.npy')
    c = np.load(save_dir + 'X_test.npy')
    d = np.load(save_dir + 'y_test.npy')
    print(a.shape, b.shape, c.shape, d.shape)
    # img = cv2.imread('/home/mt/Desktop/For_github/computer_vision_projects/sdoku_solver_with_CV/test_img/sudoku_example.jpg', 0)
    # img1 = img[73:547, 522:996]
    # img2 = img[73:547, 26:500]
    # # img = cv2.bitwise_not(img)
    # img1 = cv2.resize(img1, (28*9, 28*9))
    # img2 = cv2.resize(img2, (28*9, 28*9))
    
    # H, W = img1.shape
    # h, w = int(1/9*H), int(1/9*W)
    # # img1[(row*w):(row*w+w), (col*h):(col*h+h)]
    # row_col = [(0, 2), (0, 3), (0, 5), (0, 7), 
    #            (1, 0), (1, 3), (1, 4), (1, 6), (1, 7),
    #            (2, 0), (2, 1), (2, 2), (2, 4), (2, 5), (2, 8),
    #            (3, 1), (3, 2), (3, 4), (3, 6), (3, 8),
    #            (4, 0), (4, 1), (4, 3), (4, 4), (4, 5), (4, 7),
    #            (5, 1), (5, 3), (5, 6), (5, 8),
    #            (6, 0), (6, 2), (6, 3), (6, 4), (6, 5), (6, 8),
    #            (7, 1), (7, 3), (7, 4), (7, 6), (7, 7), 
    #            (8, 0), (8, 2), (8, 3), (8, 5), (8, 7)]
    # for index in range(len(row_col)):
    #     row, col = row_col[index]
    #     print(row, col)
    #     sub_img = img2[row*w:row*w+w, col*h:col*h+h]
    #     plt.imshow(sub_img, cmap='gray', vmin=0, vmax=255)
    #     plt.show()
        
        

