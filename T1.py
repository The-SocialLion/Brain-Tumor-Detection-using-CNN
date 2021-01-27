from keras.models import model_from_json # used to import model
from keras.models import load_model
import numpy as np
from keras.preprocessing import image# used for preproccesing 
model=load_model('SKD.h5')
print("loaded model from disk")
#classification of images
def classify(img_file):
    img_name=img_file
    test_image=image.load_img(img_name,target_size=(64,64))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    result=model.predict(test_image)

    if result[0][0]==1:
        prediction='Yes'
    else:
        prediction='No'
    print("In this{0}the results is{1}!".format(img_name,prediction))
#storing the images in this folder
import os
path='D:/python/dl programs/Brain Tumor Detection using CNN/Util/test'
files=[]
# r=root,d=directories,f=files
for r,d,f in os.walk(path):
    for file in f:
        if '.jpeg' or '.jpg' or '.png' or '.JPEG' in file:
            files.append(os.path.join(r,file))
for f in files:
    classify(f)
    print('\n')
