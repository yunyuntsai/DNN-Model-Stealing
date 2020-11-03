import sys
sys.path.insert(1, "/usr/lib/python3/dist-packages")
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2
import caffe
from caffe.proto import caffe_pb2
import lmdb

def ComputeMean(dir_path,mean_file_write_to):
    #resize 尺寸
    protosize=(48,48)

    #可以限定生成均值图像使用的图像数量
    mean_count=100
    i=0
    totalMean=np.zeros((protosize[0],protosize[1],3), dtype=np.float)
    accedImage=np.zeros((protosize[0],protosize[1],3), dtype=np.float)
    with open(dir_path,"r") as f:
        reader = csv.reader(f)
        for row in reader:
            if(i < mean_count):
                img_path= row[0]
                img_data=cv2.imread(img_path)
                img_resized=cv2.resize(img_data,protosize,interpolation=cv2.INTER_LINEAR)
                cv2.accumulate(img_resized,accedImage)

                #累计1000次计算一次均值速度会快一些，如果图像太多汇总起来再计算可能会溢出。
                if(i%10 ==0 and i>0):
                     accedImage=accedImage/float(mean_count)
                     cv2.accumulate(accedImage,totalMean)
                     accedImage=np.zeros((protosize[0],protosize[1],3),  dtype=np.float)
                print( "processed: "+str(i))
                if i==mean_count:
                    break
                i=i+1
        accedImage=accedImage/float(mean_count)
        cv2.accumulate(accedImage,totalMean)


     #for RGB image
    # totalMean=totalMean.transpose((2, 0, 1))

    # 存储为binaryproto
    blob = caffe_pb2.BlobProto()
    blob.channels=3
    blob.height = protosize[0]
    blob.width = protosize[1]
    blob.num=1
    blob.data.extend(totalMean.astype(float).flat)
    binaryproto_file = open(mean_file_write_to, 'wb' )
    binaryproto_file.write(blob.SerializeToString())
    binaryproto_file.close()

N = 100

# Let's pretend this is interesting data
X = np.zeros((N, 3, 48, 48), dtype=np.float)
Y = np.zeros(N, dtype=np.int)

dir_path = 'caffedata/Caffe_testing_100.csv'
i=0
with open(dir_path, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if(i < N):
            images_plt = plt.imread(row[0])
            # convert your lists into a numpy array of size (N, H, W, C)
            im2arr = np.array(images_plt)
            print(im2arr.shape)
            nchannels = 3
            new_img = np.resize(im2arr,(nchannels,im2arr.shape[0],im2arr.shape[1]))

            X[i] = new_img
            Y[i] = int(row[1])
            i = i+1
            print("-------------------------------------------------------------------------")

print(X)
print(Y)

map_size = X.nbytes * 10

env = lmdb.open('test_100_lmdb', map_size=map_size)

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[1]
        datum.width = X.shape[2]
        datum.height = X.shape[3]
        datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(Y[i])
        str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'),datum.SerializeToString())
ComputeMean(dir_path,'test_100_mean.binaryproto')
# train_file = '/home/yunyuntsai/Documents/Caffe/vgg19_train_val_deepid_mcv_traffic.prototxt'
# caffe_model = '/home/yunyuntsai/Documents/Caffe/VGG_ILSVRC_19_layers.caffemodel'
#
# net = caffe.Net(train_file,caffe_model,caffe.TRAIN)