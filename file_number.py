import os
path = '/home/user/dataset/VOC2007/JPEGImages'      # 输入文件夹地址
files = os.listdir(path)   # 读入文件夹
num_png = len(files)       # 统计文件夹中的文件个数
print(num_png)             # 打印文件个数
