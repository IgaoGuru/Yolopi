import cv2
import numpy as np
from torchvision import transforms
import datasets
from torch.autograd import Variable
from torch import Tensor

def get_camera_array(verbose=False, max_num_cams_to_search=10):    
    cam_array = []
    curr_cam_id = 1
    while curr_cam_id < max_num_cams_to_search:
        cap = cv2.VideoCapture(curr_cam_id)
        if cap.read()[0]:
            cam_array.append(curr_cam_id)
            curr_cam_id += 1
        else:
            break
        cap.release()        
    if verbose:
        print("Found {} camera(s)".format(len(cam_array)))
    return cam_array

def resize_frame(frame, new_width, new_height):
    # new_size should be (new_height, new_width)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    assert(frame_width >= new_width)
    assert(frame_height >= new_height)
    width_prop = new_width/frame_width
    height_prop = new_height/frame_height    
    return cv2.resize(frame, (0, 0), fx=width_prop, fy=height_prop)

def webcam_frame_to_torch_batch(webcam_frame, BGR_TO_RGB=False, pad_to_square=False, img_size=None):
    if BGR_TO_RGB:
        webcam_frame = np.copy(webcam_frame[:, :, ::-1])
    # Extract image as PyTorch tensor
    img = transforms.ToTensor()(webcam_frame)
    # Pad to square resolution
    if pad_to_square:
        img, _ = datasets.pad_to_square(img, 0)
    # Resize
    if img_size is not None:
        img = datasets.resize(img, img_size)
    # add singleton dimension at 0 to create a batch out of single image
    img_batch = img.unsqueeze(0)
    img_batch = Variable(img_batch.type(Tensor))
    return img_batch