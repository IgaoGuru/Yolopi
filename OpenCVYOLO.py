import cv2
import time

from models import *
import utils

from yolo_cam import *


USE_GPU = False

# Checks GPU availability
if USE_GPU:
    assert(torch.cuda.is_available())
num_gpus = torch.cuda.device_count()
print("Number of GPUs: {}".format(num_gpus))
for i in range(num_gpus):
    print("GPU #{}: {}".format(i, torch.cuda.get_device_name(i)))

# Defaults to CPU if GPU is not available
device = torch.device("cuda" if (torch.cuda.is_available() and USE_GPU) else "cpu")
print("Device to be used: {}".format(device))

# OpenCV variables
# Choose camera
cam_id = 0
# Speech variables
with_speech = False

cam_array = cam_util.get_camera_array(verbose=True)

# YOLO paths and configuration variables
model_def = "config/yolov3.cfg"
weights_path = "weights/yolov3.weights"
class_path = "data/coco.names"
# img size on which to run inference
img_size = 416 # should be 416 as this is the size YOLO was trained on
# confidence threshold for class consideration
conf_thres = 0.9
nms_thres = 0.5

# Get COCO classes
classes = utils.load_classes(class_path)
print(np.sort(classes))

# Load model from file
# Initiate model
model = Darknet(model_def).to(device)
if weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(weights_path)
else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(weights_path))
# Put model on desired device
model = model.to(device)
# Put model in evaluation mode
model.eval()
print("Model loaded from: {}".format(weights_path))

# Possible bounding-box colors
cmap = plt.get_cmap("tab20b")
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
# Interval on which to run YOLO inference
yolo_frame_interval = 10
# Get a reference to a video capturer
video_capture = cv2.VideoCapture(cam_array[cam_id])
#video_capture2 = cv2.VideoCapture(cam_array[1])
# loop until ESC key is pressed
frame_idx = 0
detections = None
detections2 = None
while not cv2.waitKey(33) == 27:
    frame_idx += 1
    if frame_idx > 1000:
        frame_idx = 1
    # Grab a single frame of video
    _, frame_webcam = video_capture.read()
    #_, frame_webcam2 = video_capture2.read()
    # predict for current downsized frame
    start = time.time()
    # Only run inference if we are at the right interval
    if frame_idx % yolo_frame_interval == 0:
        detections, unique_labels, bbox_colors = run_YOLO_on_frame(frame_webcam, model, img_size, device, colors=colors, conf_thres=conf_thres)
        #detections2, unique_labels2, bbox_colors2 = run_YOLO_on_frame(frame_webcam2, model, colors=colors)
    # Calculate elapsed time (also in FPS)
    end = time.time()
    elapsed_sec = end - start
    if elapsed_sec < 0.0417:
        fps = 'MAX'
    else:
        fps = str(int(1/elapsed_sec))
    # Display frame inference time
    time_lapse_label = 'Inference time: {:.0f} ms or {} fps'.format(elapsed_sec*1000, fps)
    cv2.putText(frame_webcam, time_lapse_label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    # Display latest inferred detections
    if detections is not None:
        show_YOLO_detections_on_frame(frame_webcam, detections, bbox_colors, unique_labels, classes)
    #if detections2 is not None:
    #    show_YOLO_detections_on_frame(frame_webcam2, detections2, bbox_colors2, unique_labels2, speech_engine=speech_engine)
    # Display the resulting image
    cv2.imshow('OPenCV YOLO 1', frame_webcam)
    #cv2.imshow('OPenCV YOLO 2', frame_webcam2)
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()



