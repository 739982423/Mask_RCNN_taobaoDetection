"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath("../")    #返回上上级目录的绝对路径

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library //增加一个python解析器搜索的目录
from mrcnn.config import Config
from mrcnn import model as modellib, utils          #model中定义了M-RCNN网络的结构，使用tensorflow+keras

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5") #os.path.join路径拼接，返回一个coco_weight的位置

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class BalloonConfig(Config):        #配置类，继承自Config基类，其中保存了训练推理时的超参数，可以通过继承的方式修改
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "balloon"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 25 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 150

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.1


############################################################
#  Dataset
############################################################

class BalloonDataset(utils.Dataset):        #数据读取类

    def load_balloon(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("balloon", 1, "01")
        self.add_class("balloon", 2, "02")
        self.add_class("balloon", 3, "03")
        self.add_class("balloon", 4, "04")
        self.add_class("balloon", 5, "05")
        self.add_class("balloon", 6, "06")
        self.add_class("balloon", 7, "07")
        self.add_class("balloon", 8, "08")
        self.add_class("balloon", 9, "09")
        self.add_class("balloon", 10, "10")
        self.add_class("balloon", 11, "11")
        self.add_class("balloon", 12, "12")
        self.add_class("balloon", 13, "13")
        self.add_class("balloon", 14, "14")
        self.add_class("balloon", 15, "15")
        self.add_class("balloon", 16, "16")
        self.add_class("balloon", 17, "17")
        self.add_class("balloon", 18, "18")
        self.add_class("balloon", 19, "19")
        self.add_class("balloon", 20, "20")
        self.add_class("balloon", 21, "21")
        self.add_class("balloon", 22, "22")
        self.add_class("balloon", 23, "23")
        self.add_class("balloon", 24, "24")
        self.add_class("balloon", 25, "25")


        # Train or validation dataset?
        assert subset in ["train", "val"]                   #subset是调用load_balloon时填入的参数，train或者val，如果填的不是这俩
                                                            #则subset in ["train", "val"]为false，触发异常
        dataset_dir = os.path.join(dataset_dir, subset)     #路径拼接，将数据集地址与train或val拼接

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.


        #--------------------TEST------------------------
        type_dic = {20000101:"01",20000201:"02",20000301:"03",20000401:"04",20000501:"05",20000601:"06",20000701:"07",
                    20000801:"08",20000901:"09",20001001:"10",20001101:"11",20001201:"12",20001301:"13",20001401:"14",
                    20001501:"15",20001601:"16",20001701:"17",20001801:"18",20001901:"19",20002001:"20",20002101:"21",
                    20002201:"22",20002301:"23",20002401:"24",20002501:"25",20001302:"13"}

        #——————————————————这里填video_annotation文件夹所在目录，最后记得要加\————————————————
        jsonfile_path = 'G:\python_files\\new_Mask_RCNN\Mask_RCNN_taobaoDetection\Live_dataset\\video_annotation\\'
        #————————————————————————————————————————————————————————————————————————————————

        for i in range(1, 26):
            if i < 10:
                json_name = jsonfile_path + '00000' + str(i) + '.json'
            else:
                json_name = jsonfile_path + '0000' + str(i) + '.json'
            annotations = json.load(open(json_name))
            annotations = list(annotations.values())
            for j in range(10):
                for val in annotations[1][j]['annotations']:
                    if val['instance_id'] != 0:
                        type_ = type_dic[val['instance_id']]
                        temp = val['box']
                        temp[2] = temp[2] - temp[0]
                        temp[3] = temp[3] - temp[1]


                        rects = {'x':int(temp[0]),'y':int(temp[1]),'width':int(temp[2]),'height':int(temp[3])}
                        #print(rects)
                        #print(val['instance_id'], val['box'])
                        rects = [rects]
                        #print(type_)
                        type_ = [str(type_)]
                image_name = str(i) + '_' + str(j * 40) + '.jpg'

                #———————————————这里填各个vedio的各帧图片的保存位置(每个vedio的帧都保存在这个文件夹里）————————————————
                image_path = 'G:\python_files\\new_Mask_RCNN\Mask_RCNN_taobaoDetection\Live_dataset\\video\image_prepare\\' + str(
                    i) + '_' + str(
                    j * 40) + '.jpg'
                #————————————————————————————————————————————————————————————————————————————————————————————

                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
                print(image_name)
                self.add_image(
                    "balloon",
                    image_id=image_name,  # use file name as a unique image id
                    path=image_path,
                    class_id=type_,
                    width=width, height=height,
                    polygons=rects)

        #-------------------------------------------------
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json"))) #这里读取json文件，文件名需要修改一下
        annotations = list(annotations.values())  # don't need the dict keys        #这里清除了json字典的key，我们也许需要注释掉这一步

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]              #if a['regions']为false的元素被忽略掉，也许需要注释掉这一步

        # Add images
        for a in annotations:
            rects = [r['shape_attributes'] for r in a['regions']]
            type__ = [r['region_attributes']['type'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])           #图片路径，由数据集地址+json文件中的filename项拼接而成
            image = skimage.io.imread(image_path)                           #函数作用是从文件目录下加载图片,返回numpy.ndarray对象，                                   #通道顺序为RGB，通道值默认范围0-255。
            height, width = image.shape[:2]                                 #获得.shape函数返回的list的前两列，即图片的长和宽(通道数在第三维)
            print(a['filename'])
            self.add_image(
                "balloon",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                class_id=type__,
                width=width, height=height,
                polygons=rects)

    def load_mask(self, image_id):                  #该函数在model中被调用，传入参数只有image_id
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.         如果不是气球数据集图像，则委托给父类。
        image_info = self.image_info[image_id]
        if image_info["source"] != "balloon":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape                        将多边形转换为形状的位图掩码
        # [height, width, instance_count]
        info = self.image_info[image_id]

        #class_id = self.image_info['class_id']

        class_id = np.array(image_info['class_id'], dtype=np.int32)

        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            a = p['y']
            b = p['x']
            c = p['height']
            d = p['width']
            #这里判断的目的是有些json文件里某一帧的box的值超出了图像的分辨率，比如图像是720*1280，而box的某个顶点的x却超过了宽度720或y超过了1280
            #即官方给的标注数据是有一定问题的
            if b + d >= 720:
                d = 719 - b
            if a + c >= 1280:
                c = 1279 - a
            #rr, cc = skimage.draw.rectangle((p['y'], p['x']), extent=(p['height'], p['width']))
            rr, cc = skimage.draw.rectangle((a, b), extent=(c, d))
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        return mask.astype(np.bool),class_id

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = BalloonDataset()                        #这里通过Balloon类生成数据集，然后在199行的model.train中传给model.py的train函数
    dataset_train.load_balloon(args.dataset, "train")       #在model.py的train函数中以以下的形式又返回了本文件的BalloonDataset类,并
                                                            #调用了load_mask,load_image函数来真正读取文件和目标位置:
    dataset_train.prepare()                                 #model.py.train()-> train_generator -> data_generator
                                                            #data_generator -> load_image_gt -> ballon.py.BalloonDataset() ->
                                                            #load_mask(),load_image()
    # Validation dataset
    dataset_val = BalloonDataset()
    dataset_val.load_balloon(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,                 #还是调用的model.py中Mask_RCNN类的方法train()
                learning_rate=config.LEARNING_RATE,
                epochs=175,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################
#  2021.05.20
#  python my_test.py train --dataset=G:\python_files\new_Mask_RCNN\Mask_RCNN_taobaoDetection\Live_dataset\image
#  --weights=G:\python_files\new_Mask_RCNN\Mask_RCNN_taobaoDetection\mask_rcnn_balloon_0150.h5
############################################################
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BalloonConfig()
    else:
        class InferenceConfig(BalloonConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
