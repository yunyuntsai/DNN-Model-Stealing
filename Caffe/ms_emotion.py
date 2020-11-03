
import requests
import json
import sys
import os
from os.path import join
import time
import csv
from decimal import *
import operator
from io import BytesIO

KEY = 'c8e74a1743b14e739897bacd84f7f0d5'

FACE_API_URL = 'https://southeastasia.api.cognitive.microsoft.com'  # Replace with your regional Base URL

img_url = 'ferdata/FER2013Valid/fer0028737.png'



headers = {
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': KEY,
}

params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
}

label = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown']

#body = {'url': 'http://d.ibtimes.co.uk/en/full/1571929/donald-trump.jpg'}
error = []

def annotate_image(image_path):
    # Read file
    # imgs = os.listdir(dir_path)
    # imgs.sort()
    # for img in imgs:
        dir_path = 'ferdata'
        image_path = join (dir_path, image_path)
        print("Image Path: ", image_path)
        with open(image_path,'rb') as f:
            data = f.read()

        try:
            response = requests.request('POST',FACE_API_URL + '/face/v1.0/detect',data=data,headers=headers,params=params)
            # response = requests.post(FACE_API_URL,params=params,headers=headers,json={"url": img_url})
            faces = response.json()[0]
            # print(faces)
            print('Response:')

            parsed = json.loads(response.text)[0]
            print('faceId: ',parsed['faceId'])
            print('faceRectangle: ',parsed['faceRectangle'])
            print('faceEmotion: ',parsed['faceAttributes']['emotion'])
            foo = {}
            foo.update({'neutral': float(parsed['faceAttributes']['emotion']['neutral']) })
            foo.update({'happiness': float(parsed['faceAttributes']['emotion']['happiness']) })
            foo.update({'surprise': float(parsed['faceAttributes']['emotion']['surprise']) })
            foo.update({'sadness': float(parsed['faceAttributes']['emotion']['sadness'])})
            foo.update({'anger': float(parsed['faceAttributes']['emotion']['anger'])})
            foo.update({'disgust': float(parsed['faceAttributes']['emotion']['disgust'])})
            foo.update({'fear': float(parsed['faceAttributes']['emotion']['fear'])})
            foo.update({'contempt': float(parsed['faceAttributes']['emotion']['contempt'])})
            key = max(foo.items(),key=operator.itemgetter(1))[0]
            value = Decimal(foo[key])
            print("key: ",key)
            print("value: ", value)
            print("label: ", label.index(key))
            tag = [image_path,label.index(key)]
            print (tag)
            with open("caffedata/Caffe_testing_100.csv","a") as f:
                for t in tag:
                    f.write(str(t) + ", ")
                f.write("\n")
                f.close()

            #print(json.dumps(parsed,sort_keys=True,indent=2))


        except Exception as e:

            print('Error: ', e)
            error.append(image_path)
            tag = [image_path, label.index('unknown')]
            print(tag)
            with open("caffedata/Caffe_testing_100.csv","a") as f:
                for t in tag:
                    f.write(str(t) + ", ")
                f.write("\n")
                f.close()
        time.sleep(15)
        # image_file = BytesIO(requests.get(image_url).content)
        # image = Image.open(image_file)
        #
        # plt.figure(figsize=(8,8))
        # ax = plt.imshow(image, alpha=0.6)
        # for face in faces:
        #     fr = face["faceRectangle"]
        #     fa = face["faceAttributes"]
        #     origin = (fr["left"], fr["top"])
        #     p = patches.Rectangle(origin, fr["width"], \
        #                           fr["height"], fill=False, linewidth=2, color='b')
        #     ax.axes.add_patch(p)
        #     plt.text(origin[0], origin[1], "%s, %d"%(fa["gender"].capitalize(), fa["age"]), \
        #              fontsize=20, weight="bold", va="bottom")
        # plt.axis("off")


dir_path = sys.argv[1]
count = sys.argv[2]
i=0
with open(dir_path, "r") as f:
    print("count: ", int(count))
    reader = csv.reader(f)

    for row in reader:
        if(i<int(count)):
            print("num: ", i+1 )
            annotate_image(row[0])
            i = i+1
            print("-------------------------------------------------------------------------")
print("error file: ")
print(error)







