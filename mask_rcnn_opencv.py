import cv2
import numpy as np

# load mask r-cnn
net = cv2.dnn.readNetFromTensorflow("frozen_inference_graph_coco.pb", "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

# generate random color
colors = np.random.randint(0, 255, (80, 3))
print(colors)

# load image
# img = cv2.imread("testinmultiple.jpeg")
# img = cv2.imread("testing.jpeg")
img = cv2.imread("testing2.jpeg")
# img = cv2.imread("testing_3.jpeg")
height, width, _ = img.shape

# create black image
black_image = np.zeros((height, width, 3), np.uint8)
black_image[:] = (100, 100, 0)

# detect object
blob = cv2.dnn.blobFromImage(img, swapRB=True)
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

    roi = black_image[y: y2, x: x2]
    roi_height, roi_width, _ = roi.shape

    # get the mask
    mask = masks[i, int(class_id)]
    mask = cv2.resize(mask, (roi_width, roi_height))
    _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

    if class_id < len(colors):
        color = colors[int(class_id)]
        cv2.rectangle(img, (x, y), (x2, y2), tuple([int(c) for c in color]) + (0,), 3)

    # cv2.rectangle(img, (x, y), (x2, y2), tuple((int(c) for c in colors[int(class_id)]) + [0]), 3)
    # cv2.rectangle(img, (x, y), (x2, y2), tuple(int(c) for c in colors[int(class_id)]) + (0,), 3)
    # cv2.rectangle(mask, (0, int(height)), (int(img.shape[1]), int(height)), (0, 255, 0), 5)

    # set mask coordinates
    contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow("Mask R-CNN Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
