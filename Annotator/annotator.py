import cv2
import sys
import os

import sys
sys.path.insert(0, 'Yolo')
from yolo import use_yolo


path = "Annotator/Videos/"  # set path for video directory

selection = []


def check_file(vid, class_name):
    try:
        with open(path+class_name+"_log.txt", "r") as f:
            for line in f:
                line = line.split('\n')[0]
                if line == vid:
                    return True
    except:
        return False
    return False


def on_Mouse_Event(event, x, y, flags, param):
    global selection
    if event == cv2.EVENT_LBUTTONDOWN:
        selection.append([x, y])


def assign_boxes(bb):
    res = []
    for sel in selection:
        rec = "-"
        for box in bb:
            if sel[0] > box[0] and sel[0] < (box[0] + box[2]) and sel[1] > box[1] and sel[1] < (box[1] + box[3]):
                rec = box[4]
        res.append(rec)
    print(res)
    return res


def driver(class_name):
    global selection
    files = os.listdir(path+class_name)
    for vid in files:
        bool_chk = check_file(vid, class_name)
        if bool_chk == True:
            continue
        res = {}
        cap = cv2.VideoCapture(path+class_name+"/"+vid)
        frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cnt = 0
        for i in range(frame_length):
            cnt += 1
            if cnt % 10 != 0:
                continue
            selection = []  # make sure it is global
            chk, frame = cap.read()
            cv2.namedWindow("image")
            cv2.setMouseCallback("image", on_Mouse_Event)
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            bb = use_yolo(frame)
            for box in bb:
                cv2.rectangle(
                    frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 1)
                cv2.putText(frame, box[4], (box[0], box[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("image", frame)
            k = cv2.waitKey(100000)
            temp_rec = []
            if k == ord('o'):
                temp_rec.append("out")
                temp_rec.extend(assign_boxes(bb))
                res[i] = temp_rec
                continue
            elif k == ord('i'):
                temp_rec.append("in")
                temp_rec.extend(assign_boxes(bb))
                res[i] = temp_rec
                continue
                # cv2.destroyAllWindows()
            else:
                continue
        with open(path+"Roles/"+vid+".txt", 'w') as f:
            for item in res.keys():
                feature = str(item) + ","
                for i in res[item]:
                    feature += i + ","
                f.write(feature+"\n")

        with open(path+class_name+"_log.txt", "a") as f:
            f.write(vid+'\n')
        cap.release()


class_name = sys.argv[1]
driver(class_name)
