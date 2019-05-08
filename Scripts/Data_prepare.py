import numpy as np
import os
import cv2

annotation_path = "/tmp/virat_annotations/"
video_path = "/tmp/Virat_Trimed/"
out_path = "/tmp/Data/ggnn_input/"

files = os.listdir(annotation_path)
files = [x.split(".")[0] for x in files]
files = list(set(files))


def extract_frame_data(name, ev_meta):
    frame_data = []
    vid_files = os.listdir(video_path)
    vid_name = ""
    for item in vid_files:
        data = item.split("_")
        if data[0] == ev_meta[1] and "_".join(data[1:-4]) == name and data[-3] == ev_meta[3] and data[-2] == ev_meta[4]:
            vid_name = item
            break
    '''
    cap = cv2.VideoCapture(video_path+vid_name)
    while(True):
        ret, frame = cap.read()
        frame_data.append(frame)
    '''
    return np.array([vid_name])


def extract_object_info(name, object_encoder):
    object_id = [object_encoder.index(x) for x in object_encoder if x == '1']
    object_type = []
    with open(annotation_path+name+".viratdata.objects.txt", "r") as f:
        for line in f:
            data = line.strip().split(" ")
            if int(data[0]) in object_id:
                object_type.append(data[7])
                object_id.remove(int(data[0]))
    return object_type


def driver():
    cnt = 0
    for name in files:
        with open(annotation_path+name+".viratdata.mapping.txt", "r") as f:
            for line in f:
                data = line.strip().split(" ")
                ev_meta = data[:5]
                object_encoder = data[7:]
                objects = extract_object_info(name, object_encoder)
                frame_data = extract_frame_data(name, ev_meta)
                np.savez(out_path+ev_meta[1]+"_"+str(cnt),
                         np.array(objects), frame_data)
                cnt += 1


if __name__ == "__main__":
    driver()
