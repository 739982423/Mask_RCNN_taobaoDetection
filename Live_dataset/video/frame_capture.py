
import cv2 as cv

for i in range(3,26):

    if i < 10:
        vedio_name = '00000'+str(i)+'.mp4'
    else:
        vedio_name = '0000'+str(i)+'.mp4'

    cap = cv.VideoCapture(vedio_name)

    frame_id = 0
    while(True):
        success, frame = cap.read()
        folder_name = str(i)+'\\'
        image_name = str(frame_id)+'.jpg'
        path = folder_name + image_name
        frame_id += 1
        if not success:
            break

        print(path)
        cv.imwrite(path, frame)



        #cap.set(cv.CAP_PROP_POS_FRAMES, 0)  # 设置要获取的帧号

    # 通过imwrite函数写入到文件夹
