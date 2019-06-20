import cam_util
import utils
import random
import numpy as np
import cv2

def run_YOLO_on_frame(frame_webcam, model, img_size, device, conf_thres=0.9, nms_thres=0.5, colors=None,):
    detections = None
    unique_labels = None
    bbox_colors = None
    if colors is None:
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    # Convert webcam frame to Torch batch
    img_batch = cam_util.webcam_frame_to_torch_batch(frame_webcam, BGR_TO_RGB=True, pad_to_square=True, img_size=img_size)
    # Throw image on chosen device
    img_batch = img_batch.to(device)
    # Perform inference on network
    outputs = model(img_batch)
    # Eliminate low confidence detections
    detections = utils.non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
    # Detections is a list, but we know it only contains detections for one input image
    detections = detections[0]
    if detections is not None:
        # Rescale bounding boxes to current frame img size
        detections = utils.rescale_boxes(detections, img_size, frame_webcam.shape[:2])
        # Get uniquely predicted labels
        unique_labels = detections[:, -1].cpu().unique()
        # Get number of predicted classes
        n_cls_preds = len(unique_labels)
        # Get bounding boxes' colors
        bbox_colors = random.sample(colors, n_cls_preds)
    return detections, unique_labels, bbox_colors

def show_YOLO_detections_on_frame(frame, detections, bbox_colors, unique_labels, classes):
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        cls_str = classes[int(cls_pred)]
        # Get bbox coordinates, width and height
        left = int(x1.item())
        bottom = int(y1.item())
        right = int(x2.item() )
        top = int(y2.item())
        box_w = right - left
        box_h = top - bottom
        # Get bbox color
        bbox_label = "{} {:.2f}".format(cls_str, cls_conf.item())
        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        cv2.putText(frame, bbox_label, (left, bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
        cv2.rectangle(frame, (left, bottom), (right, top), (0, 0, 255), 2)