import os
import shutil
current_path = os.path.abspath('')
for i in range(2,26):
    #os.makedirs(current_path+'\\'+'image_prepare'+'\\'+str(i))

    image_path = current_path +'\\' + str(i)

    for frame in range(0,400,40):
        origin_name = image_path+'\\'+str(frame)+'.jpg'
        target_name = str(i)+'_'+str(frame)+'.jpg'
        target_path = current_path + '\\' + 'image_prepare' + '\\'
        os.rename(origin_name,target_name)
        shutil.copy(target_name,target_path)


