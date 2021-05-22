import json
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")    #返回上上级目录的绝对路径

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library //增加一个python解析器搜索的目录


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5") #os.path.join路径拼接，返回一个coco_weight的位置

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
annotations = json.load(open(os.path.join(ROOT_DIR, "via_region_data.json"))) #这里读取json文件，文件名需要修改一下
annotations = list(annotations.values())  # don't need the dict keys        #这里清除了json字典的key，我们也许需要注释掉这一步

print(annotations)
for a in annotations:
    rects = [r['shape_attributes'] for r in a['regions']]
    print(rects)
    type = [r['region_attributes']['type'] for r in a['regions']]
    print(type)
# The VIA tool saves images in the JSON even if they don't have any
# annotations. Skip unannotated images.
# annotations = [a for a in annotations if a['regions']]