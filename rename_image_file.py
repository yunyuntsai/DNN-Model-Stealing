import os
import csv

def rename(file_path,csv_path):
#path='/home/exx/Documents/MCV_Fake_Dataset/Ad_Images_Traffic/'
    path=file_path
    n=0
    f=os.listdir(path)
    query_file_name = csv_path
    #gtFile = open('MCV_Traffic_Query.csv') # annotations file
    gtFile = open(query_file_name)
    gtReader = csv.reader(gtFile, delimiter=',') # csv parser for annotations file
    #gtReader.next() # skip header
    # loop over all images in current annotations file
    num = 0
    for row in gtReader:
        labels = []
        ad_images_name = row[0] + '.png'# the 1th column is the filename
    #images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
        labels.append(row[1]) # the 2th column is the label

    # Generate Adversarial Examples
        i=[]
        for i in f:
       #print i
           if i==ad_images_name:
              oldname=path+i
              newname=path+'class%d_'%int(labels[0]) + '%d'%int(num) + '_' + row[0] + '.png'
              os.rename(oldname,newname)
              print(oldname,'======>',newname)
              num = num + 1
           else:
              x=0
    gtFile.close()



