import os
import cv2
import argparse
from PIL import Image  

cap = cv2.VideoCapture(1)
parser = argparse.ArgumentParser(description='Save face image')
parser.add_argument('person_name', type=str, help='Name of person')
parser.add_argument('ratio', type=int, help='ratio of the frame size')
args = parser.parse_args()

def main(person_name, ratio):
    # get PWD
    full_path = os.path.realpath(__file__)
    save_dir = os.path.dirname(full_path) + '/' + person_name + '/'  # /home/mt/Desktop/For_github/computer_vision_projects/face_recognition
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    start_index = 1
    while True:
        # print(save_dir)
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64*ratio, 48*ratio))
            cv2.imshow('gray', gray)
            key = cv2.waitKey(1)
            
            if key == 32: # space
                # save the image
                cv2.imshow('save', gray)
                img = Image.fromarray(gray)
                # create folder "person_name" if does not exist
                img = img.save(save_dir + str(start_index)+'.jpg')
                start_index += 1
                
            elif key == 27:
                break
    cap.release()   
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(args.person_name, args.ratio)