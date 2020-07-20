import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize
from model import log


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco_humanpose.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
COCO_DIR = "images"  # TODO: enter value here
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    KEYPOINT_MASK_POOL_SIZE = 7

inference_config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=inference_config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

import cv2
# COCO Class names
#For human pose task We just use "BG" and "person"
class_names = ['BG', 'person']
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
# image = cv2.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
# image = cv2.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
# image = cv2.imdecode(np.fromfile(os.path.join(IMAGE_DIR, random.choice(file_names)), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
imgs = [IMAGE_DIR + '//' + i for i in os.listdir(IMAGE_DIR) if i.endswith('.jpg')]
for img in imgs:
    image = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    #BGR->RGB
    image = image[:,:,::-1]

    # Run detection
    results = model.detect_keypoint([image], verbose=1)
    r = results[0] # for one image

    log("rois",r['rois'])
    log("keypoints",r['keypoints'])
    log("class_ids",r['class_ids'])
    log("keypoints",r['keypoints'])
    log("masks",r['masks'])
    log("scores",r['scores'])

    visualize.display_keypoints(image,r['rois'],r['keypoints'],r['class_ids'],class_names,skeleton = inference_config.LIMBS)
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])


