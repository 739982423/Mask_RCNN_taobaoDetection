import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("")
print(ROOT_DIR)
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/balloon/"))  # To find local version
import my_test

#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_balloon_addvedio0175.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(my_test.BalloonConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()



# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['bg','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17',
               '18','19','20','21','22','23','24','25']

#--------------这里的地址填图片所在位置--------------
path_traindata = "G:\python_files\\new_Mask_RCNN\Mask_RCNN_taobaoDetection\Live_dataset\image\\train\\"
path_vediodata = "G:\python_files\\new_Mask_RCNN\Mask_RCNN_taobaoDetection\Live_dataset\\video\image_prepare\\"

#一共113幅图像，连续测试
for i in range(5,15):

    #训练集图片的地址
    image = skimage.io.imread(path_traindata+str(i)+'.jpg')
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    print(str(i)+'.jpg')

    # if r['class_ids'] == []:
    #     predict = 'Unpredictable'
    # else:
    #     predict = max(r['class_ids'])

    predict = r['class_ids']
    print('图片类别是{},概率为{}'.format(predict,r['scores']))
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                  class_names, r['scores'])

#一共113幅图像，连续测试
for i in range(1,26):

    #训练集图片的地址
    for j in range(0,361,40):

        image = skimage.io.imread(path_vediodata + str(i) + '_'+ str(j) +'.jpg')
        results = model.detect([image], verbose=1)

        # Visualize results
        r = results[0]

        # if r['class_ids'] == []:
        #     predict = 'Unpredictable'
        # else:
        #     predict = max(r['class_ids'])

        predict = r['class_ids']
        print(str(i) + '_' + str(j) + '.jpg')
        print('图片类别是{},概率为{}'.format(predict,r['scores']))
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                  class_names, r['scores'])


