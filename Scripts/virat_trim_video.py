
import cv2
import os

output_path = "/home/alpha/Work/Dataset/Virat_Ground/Virat_Trimed/"
annotation_path = "/home/alpha/Work/Dataset/Virat_Ground/VIRAT Ground Dataset/annotations/"
original_path = "/home/alpha/Work/Dataset/Virat_Ground/VIRAT Ground Dataset/videos_original/"

ann_files = os.listdir(annotation_path)
ann_files = [x for x in ann_files if "events" in x]


def check_file(fil_name):
    with open("Scripts/log.txt", 'r') as fl:
        for line in fl:
            line = line.split('\n')[0]
            if line == fil_name:
                return False
        return True


for item in ann_files:
    ret = check_file(item)
    if ret == False:
        continue

    records = {}
    with open(annotation_path+item, 'r') as fl:
        flag = 0
        for line in fl:
            data = line.split('\n')[0].split(" ")
            if int(data[0]) not in records.keys():
                records[int(data[0])] = [data[1],data[3],data[4]]

    vid_name = item.split('.')[0] + ".mp4"
    for key in records.keys():
        record = records[key]
        cap = cv2.VideoCapture(original_path+vid_name)
        frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        out = cv2.VideoWriter(output_path+record[0]+"_"+item.split('.')[0]+"_"+str(
            key)+".avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
        for i in range(frame_length):
            chk, frame = cap.read()
            if i >= int(record[1]) and i <= int(record[2]):
                out.write(frame)
            if i > int(record[2]):
                break
        cap.release()
    with open("Scripts/log.txt", "a") as fl:
        fl.write(item+"\n")
