from azure.cognitiveservices.vision.customvision.prediction import prediction_endpoint
from azure.cognitiveservices.vision.customvision.training import training_api
from azure.cognitiveservices.vision.customvision.prediction.prediction_endpoint import models
import sklearn.metrics
import pandas as pd
import numpy as np
import itertools
import os
import csv
from os.path import join
import glob
from matplotlib.pyplot import tight_layout, ylabel, figure, imshow, yticks, colorbar, xticks, show, xlabel, cm, text, \
    suptitle


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    imshow(cm, interpolation='nearest', cmap=cmap)
    suptitle(title, fontsize=14, horizontalalignment="right")
    colorbar()
    tick_marks = np.arange(len(classes))
    xticks(tick_marks, classes, rotation=45, horizontalalignment="right")
    yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        text(j, i, "{:0.2f}".format(cm[i, j]),
             horizontalalignment="center",
             size=8,
             color="white" if cm[i, j] > thresh else "black")

    tight_layout()
    ylabel('True label')


###CHANGE testdata path#######################
dir_src = "TestImages/traffic_testImages"

###CHANGE ProjectID######
project_Id = ""

###CHANGE output path###
outputPath = "MCV_Query.csv"
#############################################

training_key = ""
prediction_key = ""

predictor = prediction_endpoint.PredictionEndpoint(prediction_key)
print(predictor)
trainer = training_api.TrainingApi(training_key)
project = trainer.get_project(project_Id)
print(project.id)

tags = trainer.get_tags(project.id)

tag_dic = {}
tag = []
for t in tags:
    tag.append(t.name)
    tag_dic[t.name] = t.id


count = 0
test_ids = []
preds = []

test_dir = os.listdir(dir_src)
test_dir.sort()
for d in test_dir:
    d_path = join(dir_src, d)
    image_dir = os.listdir(d_path)
    for p in image_dir:

        path = join(d_path, p)
        print(path)
        with open(path, mode="rb") as test_data:
            #print(test_data)
            results = predictor.predict_image(project.id, test_data)

        # Display the results.
        re = []
        re.append(path.split('/')[2] + "_" + path.split('/')[-1][:-4])
        for i  in  range(0,5):
            predictedClass = results.predictions[i].tag_name
            predictedProb = results.predictions[i].probability
            print(predictedClass)
            print(predictedProb)

            re.append(predictedClass)
        print(re)
        with open(outputPath, "a") as f:
            for r in re:
                f.write(r+",")
            f.write("\n")

        f.close()


input("All done. Press any key...")

