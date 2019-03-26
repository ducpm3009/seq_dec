#josh.anish1@gmail.com
#the code has been created with an intention of previewing the entire project
#contributions are welcomed!!!

import cv2
import numpy as np
import os
import sys
import pandas as pd
import pprint
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


def get_clothes_category(clothes_name,file_name = "list_category_cloth.csv"):
    file = path_file(file_name)
    reader = pd.read_csv(file)
    return reader[reader[clothes_name] == clothes_name]

def path_file(file):
    return str(file)

def nn(input_dir,output_dir):

    all_dir = []
    all_file = []
    clothes_name = []

    for _dir in os.listdir(input_dir):
        all_dir.append(_dir)
        clothes_name.append(_dir.split('_')[-1])
    clothes_categories = dict(zip(clothes_name,all_dir))
    # print(list(clothes_categories.values()))
    # pprint.pprint(clothes_categories)
    # print(len(all_file) , sep="\n") 
    # print(*all_dir , sep="\n")

    for _dir in list(clothes_categories.values()):
        input_subdir = os.getcwd() + "/" + input_dir + "/" + _dir
        output_file_dir = os.getcwd() + "/" + output_dir + "/" + _dir
        
        for img_file in os.listdir(input_subdir):
            img_file = input_subdir + "/" +img_file
            predict = predictor(img_file)
            reader = pd.read_csv(path_file("list_bbox.csv"))
            img = cv2.imread(img_file)
            mask = np.zeros(img.shape[:2],np.uint8)   
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)
            
            output_file = img_file.split('/')
            sub_folder_path = output_dir + '/' + _dir
            if reader.image_name[predict]:
                rect = (reader.x1[predict],reader.y1[predict],reader.x2[predict],reader.y2[predict])
                cv2.grabCut(img,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
                mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
                
                img_cut = img*mask2[:,:,np.newaxis]
                if not os.path.exists(sub_folder_path):
                    os.makedirs(sub_folder_path)
                cv2.imwrite(os.path.join(output_file_dir,img_file),img_cut)
                # print(output_file_dir + '/' +img_file)

nn(input_dir,output_dir)
