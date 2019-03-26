#josh.anish1@gmail.com
#the code has been created with an intention of previewing the entire project
#contributions are welcomed!!!

import cv2
import numpy as np
import os
import sys
import pandas as pd
# import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Conv2D

from keras.layers import MaxPooling2D
from keras.layers import Dense

from keras.layers import Flatten,Dropout
from keras.models import model_from_json

input_dir = sys.argv[1]
output_dir = sys.argv[2]



def predictor(img_file):
    img = cv2.imread(img_file)
    resize = cv2.resize(img,(64,64))
    #resize = np.expand_dims(resize,axis=0)
    

    img_fin = np.reshape(resize,[1,64,64,3])
    json_file = open('model/binaryfas10.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model/binaryfashion.h5")
    # print("Loaded model from disk")
    
    prediction = loaded_model.predict_classes(img_fin)
    
    prediction = np.squeeze(prediction,axis=1)
    predict = np.squeeze(prediction,axis=0)
    return int(predict)

"""Neural Network Decoding"""
""" The coordinates are created and trained"""
"""-----------------"""
image_width = 300
image_height = 500



def path_file(file):
    return str(file)

def nn(input_dir,output_dir):

    all_dir = []
    all_file = []
    for _dir in os.listdir(input_dir):
        cwd = os.getcwd() + "/" + input_dir
        path = cwd  + "/" + _dir
        for file in os.listdir(path):
            all_file.append(path + "/" + file)
        all_dir.append(cwd  + "/" + _dir)
        # print(*all_file , sep="\n")
        # print(*all_dir , sep="\n")
    
    for img_file in all_file[0 : 40]:
        predict = predictor(img_file)
        file = path_file("list_bbox.csv")
        reader = pd.read_csv(file)

        img = cv2.imread(img_file)
        #img = cv2.resize(img,(image_width,image_height))
        #seg = image(image,reader.x1[predict],reader.y1[predict],reader.x2[predict],reader.y2[predict],reader.i[predict])
        

        mask = np.zeros(img.shape[:2],np.uint8)   
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        
        output_file = img_file.split('/')
        cur_img_input_path = 'img/' + output_file[6] + '/' +output_file[7]
        sub_folder_path = output_dir +'/' + output_file[6]
        if reader.image_name[predict]:
            rect = (reader.x1[predict],reader.y1[predict],reader.x2[predict],reader.y2[predict])
            cv2.grabCut(img,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
            
            img_cut = img*mask2[:,:,np.newaxis]
            if not os.path.exists(sub_folder_path):
                os.makedirs(sub_folder_path)
            image_output_path = os.getcwd()+'/' +sub_folder_path
            # print(image_output_path)
            cv2.imwrite(os.path.join(image_output_path,output_file[7]),img_cut)
            print(image_output_path + '/' +output_file[7])

nn(input_dir,output_dir)
