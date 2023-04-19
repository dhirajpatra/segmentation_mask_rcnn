import cv2
import numpy as np

# load mask r-cnn
net = cv2.dnn.readNetFromTensorflow("frozen_inference_graph_coco.pb", "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

# generate random color
colors = np.random.randint(0, 255, (80, 3))

# open video capture
cap = cv2.VideoCapture(0)

while True:
    # read frame
    ret, img = cap.read()
    if not ret:
        break

    height, width, _ = img.shape

    # create black image
    black_image = np.zeros((height, width, 3), np.uint8)
    black_image[:] = (100, 100, 0)

    # detect object
    blob = cv2.dnn.blobFromImage(img, swapRB=True)
    net.setInput(blob)

    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    detection_count = boxes.shape[2]

    object_count = 0  # initialize object count

    for i in range(detection_count):
        box = boxes[0, 0, i]
        class_id = box[1]
        score = box[2]
        if score < 0.5:
            continue

        object_count += 1  # increment object count

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

        # set mask coordinates
        mask_array = np.array(mask, np.uint8)
        contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Add the text to the image
    # Set font and text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = str(object_count) + ' persons detected'
    org = (50, 50)
    fontScale = 1.5
    color = (255, 0, 0)  # BGR color format
    thickness = 2  
    cv2.putText(img, text, org, font, fontScale, color, thickness)
    
    cv2.imshow("Mask R-CNN Output", img)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
