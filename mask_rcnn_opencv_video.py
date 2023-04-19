import cv2
import numpy as np

# load mask r-cnn
net = cv2.dnn.readNetFromTensorflow("frozen_inference_graph_coco.pb", "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

# generate random color
colors = np.random.randint(0, 255, (80, 3))
print(colors)

# open video capture
cap = cv2.VideoCapture(0)

while True:
    # read frame
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # detect object
    blob = cv2.dnn.blobFromImage(frame, swapRB=True)
    net.setInput(blob)

    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    detection_count = boxes.shape[2]

    for i in range(detection_count):
        box = boxes[0, 0, i]
        class_id = box[1]
        score = box[2]
        if score < 0.5:
            continue

        # get box coordinates
        x = int(box[3] * width)
        y = int(box[4] * height)
        x2 = int(box[5] * width)
        y2 = int(box[6] * height)

        roi = frame[y: y2, x: x2]
        roi_height, roi_width, _ = roi.shape

        # get the mask
        mask = masks[i, int(class_id)]
        mask = cv2.resize(mask, (roi_width, roi_height))
        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

        if class_id < len(colors):
            color = colors[int(class_id)]
            color = tuple(map(int, color))
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, str(class_id), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, tuple(color), 2)

        # set mask coordinates
        contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(frame, [contour + (x, y)], 0, color, 2)

    cv2.imshow("Mask R-CNN Output", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
