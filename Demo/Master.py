import os
import cv2
import time

########################################################################################
os.system('figlet VideoLSTM Started')
start = time.time()
os.system('python Demo/videolstm_demo.py')
end = time.time()
os.system('figlet Time Taken for Completion: ' + str(round((end-start), 2)) + ' seconds')
########################################################################################
os.system('figlet FCN-VGG13 Started')
start = time.time()
os.system('python Demo/fcn_demo.py')
os.system('figlet Time Taken for Completion: ' + str(round((end-start), 2)) + ' seconds')
########################################################################################
img_path = 'Demo/Data/images/'
f = os.listdir(img_path)
for i in range(5):
    p = img_path + f[i+i*5]
    img = cv2.imread(p)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
########################################################################################
os.system('figlet GGNN Started')
start = time.time()
os.system("python GGNN/train_ggnn.py")
os.system('figlet Time Taken for Completion: ' + str(round((end-start), 2)) + ' seconds')
########################################################################################