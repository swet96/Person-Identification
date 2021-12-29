""" 
Useful Functions for COL780 Assignment 3
"""

import cv2
import os, shutil
import random
import json
import numpy as np
from tqdm import tqdm
from skimage.transform import pyramid_gaussian

# Prepare Data For training SVM
def prepare_data(args,template_size):

    f=open(args.train_json,'r') 
    data=json.load(f)
    images_dict=data['images']
    images=[images_dict[i]['file_name'].split('/')[2] for i in range(len(images_dict))]
    f.close()


    print("Preparing Training Data")

    train_x = [] #trainx containes images
    train_y = [] #contains labels, 1 for postive images and 0 for negative images
    num_pos=0
    num_neg=0
    for num,file in enumerate(tqdm(images)):
        
        image = cv2.imread(os.path.join(args.inp_folder,"PNGImages",file))
        with open(os.path.join(args.inp_folder,"Annotation" ,file[:-3]+"txt")) as f:
            bbox_arr = []
            for line in f:
                if "Bounding box" in line:
                    line = line.split(":")[1].strip("\n").strip().replace(" ","")
                    coordinate_arr = line.split("-")
                    x1,y1 = map( lambda x:int(x) ,coordinate_arr[0].replace(")","").replace("(","").split(","))
                    x3,y3 = map( lambda x:int(x) ,coordinate_arr[1].replace(")","").replace("(","").split(","))
                    bbox_arr.append([x1,y1,x3,y3])

        # Take negative samples of [template_size[0],template_size[1]] -> Resize later to req. template_size
        for nothing in range(3):
            h, w = image.shape[:2]
            if h > template_size[0] or w > template_size[1]:
                h = h - template_size[0]; 
                w = w - template_size[1]

                overlap = [True for i in bbox_arr]
                max_loop = 0
                while np.any(overlap) ==True:
                    max_loop+=1
                    if max_loop==10:
                        break
                    overlap = [True for i in bbox_arr]
                    x = random.randint(0, w)
                    y = random.randint(0, h)
                    window = [x,y,x+template_size[1],y+template_size[0]]
                    for var,bbox in enumerate(bbox_arr):
                        dx = min(bbox[2], window[2]) - max(bbox [0], window[0])
                        dy = min(bbox[3], window[3]) - max(bbox[1],  window[1])
                        if dx<=0 or dy<=0:
                            overlap[var] = False
                if max_loop<10:
                    img = image[window[1]:window[3],window[0]:window[2]]
                    train_x.append(img)
                    train_y.append(0)
                    num_neg+=1

        for box in bbox_arr:
            img = image[box[1]:box[3],box[0]:box[2]]
            train_x.append(img)
            train_y.append(1)
            num_pos+=1
            
    train_x = [cv2.resize(image,(template_size[1],template_size[0])) for image in train_x]
    print(f"Prepared {num_pos} positive & {num_neg} training examples")

    if args.vis:
        print(f"Saving training images in {str(args.inp_folder)+'/train_data_hog_custom'}")
        if os.path.exists(str(args.inp_folder)+"/train_data_hog_custom"):
            shutil.rmtree(str(args.inp_folder)+"/train_data_hog_custom")
        os.mkdir(str(args.inp_folder)+"/train_data_hog_custom")
        for num_sample,img in enumerate(train_x):
            cv2.imwrite(str(args.inp_folder)+"/train_data_hog_custom/"+str(train_y[num_sample])+"_"+str(num_sample)+".png",img)

    return train_x,train_y



def NMS(boxes, confidence,th = 0.3):
    if len(boxes) == 0:
        return np.array([], dtype=int),np.array([], dtype=float)
    rects_with_confidence = [[boxes[i],confidence[i]] for i in range(len(boxes))]

    # Sort according to confidence
    rects_with_confidence = (sorted(rects_with_confidence, key=lambda box: box[1][0],reverse=True))

    rects = [var[0] for var in rects_with_confidence]
    
    bool_arr = [True for i in rects_with_confidence]
    
    for i,box in enumerate(rects):
        if bool_arr[i] == True:
            for j,other_box in enumerate(rects[i+1:]):
                k = j+i+1
                if bool_arr[k] == True:
                    dx = max(0,min(box[2], other_box[2]) - max(box [0], other_box[0]))
                    dy = max(0,min(box[3], other_box[3]) - max(box[1], other_box[1]))
                    
                    overlap = float(dx*dy)
                    overlap_percentage = overlap/((other_box[3]-other_box[1])*(other_box[2]-other_box[0]))
                    if overlap_percentage > th:
                        bool_arr[k] = False
                    
    
    final_rects = []
    final_confidence = []
    for i,rect in enumerate(rects):
        if bool_arr[i]:
            final_rects.append(rect)
            final_confidence.append(rects_with_confidence[i][1][0])
    
    return np.array(final_rects, dtype=int),np.array(final_confidence, dtype=float)




# Sliding Window
def sliding_window(image, template_size, step_size):
    res = []
    for y in range(0, image.shape[0], step_size[0]):
        for x in range(0, image.shape[1], step_size[1]):
            res.append([x, y, image[y: y + template_size[0], x: x + template_size[1]]])
    return res


