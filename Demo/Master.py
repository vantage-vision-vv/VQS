import os
import cv2
import time

########################################################################################
os.system('figlet VideoLSTM Started')
start = time.time()
os.system('python Demo/videolstm_demo.py')
end = time.time()
os.system('figlet Time Taken for Completion: ' + str(round((end-start), 2)) + ' seconds')
os.system('figlet ..............................')
########################################################################################
os.system('figlet FCN-VGG13 Started')
start = time.time()
os.system('python Demo/fcn_demo.py')
end = time.time()
os.system('figlet Time Taken for Completion: ' + str(round((end-start), 2)) + ' seconds')
os.system('figlet ..............................')
########################################################################################
os.system('figlet GGNN Started')
start = time.time()
os.system("python GGNN/train_ggnn.py")
end = time.time()
os.system('figlet Time Taken for Completion: ' + str(round((end-start), 2)) + ' seconds')
os.system('figlet ..............................')
########################################################################################