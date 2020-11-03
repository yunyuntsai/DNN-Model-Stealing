# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#            
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import matplotlib.pyplot as plt
import csv
import PIL
from PIL import Image
import optimize
import utils
import numpy as np
import os
# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns(rootpath): #/home/honggang/Documents/Steal_DL_Models/MCV_Fake_Dataset/
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    model_name = 'caffenet' #????????????????????????????????????????????????
    layer = 'fc7'
    net = utils.init_model(model_name)
    # loop over all 42 classes
    num = 0
    #for c in range(0,43):
    #prefix = rootpath + 'Source_Images_Traffic'+'/' # subdirectory for class    
    prefix = '/home/exx/Documents/dataset_all/gtsrb_2150/' #????????????????????????????????????????????????
    ad_image_save_path = '/home/exx/Documents/dataset_all/' + 'featureadversary_dataset2150' + '/' #????????????????????????????????????????????????
    gtFile = open('/home/exx/Documents/dataset_all/' + 'MCV_Traffic_Query2150.csv') # annotations file #????????????????????????????????????????????????
    gtReader = csv.reader(gtFile, delimiter=',') # csv parser for annotations file
    #gtReader.next() # skip header
    # loop over all images in current annotations file
    for row in gtReader:
        labels = []
        source_images_path = prefix + row[0] + '.jpg'# the 1th column is the filename
        #images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
        #labels.append(row[1]) # the 2th column is the label
        labels=row[2] # the 3th column is the label
        #labels.append(row[3]) # the 4th column is the label
        #labels.append(row[4]) # the 5th column is the label
        #labels.append(row[5]) # the 6th column is the label
        print source_images_path
        print labels
        #print labels[1]
        #print labels[2]
        #print labels[3]
        #print labels[4]
        
        os.system("pause");
        # Generate Adversarial Examples
        src_fname = source_images_path
        src = utils.read_image_rgb(src_fname, net)
        #for i in range(0,5):
        guide_images_path = '/home/exx/Documents/dataset_all/' + 'Guide_Images_Traffic' + '/' + labels + '/' + 'guide.jpg'
        guide_fname = guide_images_path
        guide = utils.read_image_rgb(guide_fname, net)
        iter_n=10
        max_thres=5
        ad, bc, cd = optimize.optimize_single(src_fname, guide_fname, model_name, layer, iter_n, max_thres, net)
        img = PIL.Image.fromarray(np.uint8(ad))
        img.save(ad_image_save_path + '%d'%int(num) + '_' + '%d'%int(labels) + '.jpg') #c-source label labels[i]-guide label   
        num = num + 1
            
    gtFile.close()
    #return images, labels
