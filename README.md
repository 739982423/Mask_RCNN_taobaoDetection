# Mask R-CNN for TaoBao Object Detection
根目录下的`my_test.py`为训练文件，`demo.py`为推理文件  
数据集已经存好，放在`Live_dataset`目录下，包含了训练集和测试集（这里用的测试集为训练集的子集）

**使用my_test.py进行训练:**  
```python my_test.py train --dataset=D:\python\Mask_RCNN\Mask_RCNN\Live_dataset\image --weights=coco```  
这里的`--dataset=`需要修改为本地训练图片的位置，就是这个库的`Live_dataset\image`

**使用demo.py进行推理:**  
```python demo.py```  
  
**效果如图所示:**  
![](https://github.com/739982423/Mask_RCNN_taobaoDetection/blob/master/Live_dataset/example.png)  
![](https://github.com/739982423/Mask_RCNN_taobaoDetection/blob/master/Live_dataset/example1.png)
