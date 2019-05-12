import numpy as np
import cv2
import os

###############################################################################
# SetUp YOLO
################################################################################
labelsPath = os.path.sep.join(["Yolo/yolo-coco/", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = os.path.sep.join(["Yolo/yolo-coco/", "yolov3.weights"])
configPath = os.path.sep.join(["Yolo/yolo-coco/", "yolov3.cfg"])
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def use_yolo(image):
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    ret = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            entry = []
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            entry = [x, y, w, h, LABELS[classIDs[i]]]
            ret.append(entry)
    return ret
