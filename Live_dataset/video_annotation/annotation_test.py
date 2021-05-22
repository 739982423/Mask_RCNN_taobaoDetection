import json
import skimage.draw
import skimage.io
import os
print(isinstance('1',int))
print(type([[1],[2],[3]]))
# for i in range(1,26):
#     if i < 10:
#         json_name = '00000' + str(i) + '.json'
#     else:
#         json_name = '0000' + str(i) + '.json'
#     annotations = json.load(open(json_name)) #这里读取json文件，文件名需要修改一下
#     annotations = list(annotations.values())  # don't need the dict keys        #这里清除了json字典的key，我们也许需要注释掉这一步
#     for j in range(10):
#         for val in annotations[1][j]['annotations']:
#             if val['instance_id'] != 0:
#                 print(val['instance_id'],val['box'])
#         image_path = 'D:\python\Mask_RCNN\Mask_RCNN\Live_dataset\\video\image_prepare\\' + str(i) +'_'+ str(j*40) + '.jpg'
#         image = skimage.io.imread(image_path)
#         height, width = image.shape[:2]