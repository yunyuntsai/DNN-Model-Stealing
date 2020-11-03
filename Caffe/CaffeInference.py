import sys
sys.path.insert(1, "/usr/lib/python3/dist-packages")
import caffe
import csv
import matplotlib.pyplot as plt

import numpy as np

deploy_file ='vgg19_deepid_mcv_traffic_deploy.prototxt'
caffe_model = 'model/emotion-model_iter_10000.caffemodel'
net = caffe.Net(deploy_file,caffe_model,caffe.TEST)
#net = caffe.Classifier(deploy_file, caffe_model)
#test_image = 'ferdata/FER2013Valid/fer0028657.png'

#图片预处理设置
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  #设定图片的shape格式(1,3,28,28)
transformer.set_transpose('data', (2,0,1))    #改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    #减去均值，前面训练模型时没有减均值，这儿就不用
#transformer.set_raw_scale('data', 255)    # 缩放到【0，255】之间
#transformer.set_channel_swap('data', (2,1,0))   #交换通道，将图片由RGB变为BGR

dir_path = sys.argv[1]
with open(dir_path, "r") as f:
    true_count = 0
    count = 0
    reader = csv.reader(f)
    for row in reader:
        test_image = row[0]
        true_label = row[1]
        print(test_image)
        im=caffe.io.load_image(test_image)                   #加载图片
        print(im.shape)
        input = [im]
        # Classify image
        #prediction = net.predict(input)  # predict takes any number of images, and formats them for the Caffe net automatically
        #print('predicted classes: ',prediction)
        #plt.imshow(im)
        #plt.show()
        net.blobs['data'].data[...] = transformer.preprocess('data',im)      #执行上面设置的图片预处理操作，并将图片载入到blob中
        # print(net)
        # #执行测试
        out = net.forward()
        #
        # # labels = np.loadtxt(labels_filename, str, delimiter='\t')   #读取类别名称文件
        prob= net.blobs['prob'].data[0].flatten() #取出最后一层（Softmax）属于某个类别的概率值，并打印
        # #print("prob: ",prob)
        order=prob.argsort()[-1]  #将概率值排序，取出最大值所在的序号
        print ('predict class: %i true label: %i' %(int(order), int(true_label)))   #将该序号转换成对应的类别名称，并打印
        if int(order) == int(true_label):
            	true_count = true_count+1
        count = count+1
    print("Total : %i  True: %i Acc: %.2f" % (count, true_count, float(true_count/count)))



