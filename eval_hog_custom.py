""" 
COL780
Assignment 3
"""

import cv2
import argparse
import os, shutil
import joblib
import json
import numpy as np
from tqdm import tqdm
from skimage.feature import hog
from sklearn.svm import SVC
import torch,torchvision
from utils import *


def hog_custom(args):

    template_size = [128,64] # H, W  -> Standard Size as per paper
    step_size = (10,10) # For sliding window
    confidence_threshold = 0.5 # For Positive Prediction
    scale_factor = 1.25 # For Gaussian Pyramid
    
    if os.path.exists(str(args.inp_folder)+'/saved_models/hog_custom.pkl'):
        classifier =  joblib.load(str(args.inp_folder)+"/saved_models/hog_custom.pkl")
        print(f"Trained model found at {args.inp_folder+'saved_models/hog_custom.pk'}. Loading it...")
    else:
        print(f"Trained model NOT found at {args.inp_folder+'saved_models/hog_custom.pk'}\nTraining New...")

        img,Y = prepare_data(args,template_size) #TODO
        X = []        
        # Calculating HoG Descriptors
        print("Computing HoG features: ")
        for i,x in  enumerate(tqdm(img)):
            x = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
            
            # Compute HoG feature descriptor
            # fd, hog_image = hog(x, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2',visualize=True, feature_vector=True)
            fd = hog(x, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2', feature_vector=True)
               
            """
            # Visualize HoG
            if Y[i]==1 and i%10==0:
                import matplotlib.pyplot as plt
                plt.imshow(hog_image,cmap='gray')
                plt.title("HOG")
                plt.show()
                cv2.waitKey(0)
            """
            X.append(fd)
        print(f"shape of x: {len(X[0])}")
        print(len(Y))
        
        train_x = np.array(X)
        train_y = np.array(Y)
        # Support Vector Machine for Classification
        classifier = SVC (kernel='rbf',gamma='scale')
        print("Training SVM Classifier")
        classifier.fit(train_x,train_y)
        # Save Model
        if not os.path.exists(str(args.inp_folder)+'/saved_models'):
            os.mkdir(str(args.inp_folder)+'/saved_models')
        print(f"Saved Trained Model as {args.inp_folder}+'/saved_models/hog_custom.pkl'")
        joblib.dump( classifier, str(args.inp_folder)+"/saved_models/hog_custom.pkl")

    print("Testing the model...")
    if os.path.exists(str(args.inp_folder)+"/vis_hog_custom"):
        shutil.rmtree(str(args.inp_folder)+"/vis_hog_custom")
    os.mkdir(str(args.inp_folder)+"/vis_hog_custom")

    coco_result = []
    category_id = 1
    
    f=open(args.val_json,'r') 
    data=json.load(f)
    images_dict=data['images']
    images=[images_dict[i]['file_name'].split('/')[2] for i in range(len(images_dict))]
    annotations_dict=data["annotations"]
    image_ids=[annotations_dict[i]['image_id'] for i in range(len(annotations_dict))]
    image_ids=np.unique(image_ids)
    f.close()
    
    for i in tqdm(range(len(images))):
        file=images[i]
        image_id=image_ids[i]

        rects = []
        confidence = []
        #image_id=
        
        image = cv2.imread(os.path.join(args.inp_folder,"PNGImages",file))
        
        #Original Dimensions
        h,w = image.shape[0],image.shape[1]

        original_image = image.copy()

        # Resize for speed
        image = cv2.resize(image,(400,256))
        h_ratio = h/256
        w_ratio = w/400

        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # List to store the detections
        rects = []
        confidence= []
        # For storing current scale
        scale = 0
        for im_scaled in pyramid_gaussian(image, scale_factor):
            #The list contains detections at the current scale
            if im_scaled.shape[0] > template_size[0] and im_scaled.shape[1] and template_size[1]:
                # Sliding Window for each level of pyramud
                windows = sliding_window(im_scaled, template_size, step_size)
                for (x, y, window) in windows:
                    if window.shape[0] == template_size[0] and window.shape[1] == template_size[1]:
                        fd=hog(window, orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(2,2))
                        fd = fd.reshape(1, -1)
                        pred = classifier.predict(fd)
                        if pred == 1:
                            confidence_score = classifier.decision_function(fd)
                            if confidence_score > confidence_threshold:
                                x1 = int(x * (scale_factor**scale)*w_ratio) # x_tl_in_original_image
                                y1 = int(y * (scale_factor**scale)*h_ratio) # y_tl_in_original_image
                                w_in_original_image = int(template_size[1] * (scale_factor**scale)*w_ratio)
                                h_in_original_image = int(template_size[0] * (scale_factor**scale)*h_ratio)
                                x3 = x1 + w_in_original_image # x_br_in_original_image
                                y3 = y1 + h_in_original_image # y_br_in_original_image
                                rects.append([x1,y1,x3,y3])
                                confidence.append([confidence_score])
            
                scale += 1
            else:
                break

        # Apply Non-max suppression
        rects,scores = NMS(rects,confidence)

        for rect,score in zip(rects,scores):
            x1,y1,x3,y3 = rect.tolist()
            coco_result.append({"image_id":int(image_id),"category_id":int(category_id),"bbox":[float(x1),float(y1),float(x3-x1),float(y3-y1)],"score":np.round(score.item(),3)})
            if args.vis:
                cv2.rectangle(original_image, (x1, y1), (x3, y3), (0, 0, 255), 2)
        if args.vis:
            cv2.imwrite(str(args.inp_folder)+"/vis_hog_custom/"+str(file),original_image)
        
    print(f"Saved predictions at {args.inp_folder+'/pred_eval_hog_custom.json'}")
    json.dump(coco_result, open(args.inp_folder+"/pred_eval_hog_custom.json", 'w'), ensure_ascii=False)


def main(args):
    hog_custom(args)

if __name__ == "__main__":
    argument_parser_object = argparse.ArgumentParser(description="Pedestrian Detection in images")
    argument_parser_object.add_argument('-i', '--inp_folder', type=str, default='PennFudanPed', help="Path for the root folder of dataset containing images, annotations etc.)")
    argument_parser_object.add_argument('-v', '--vis', action='store_true', default=False, help="Visualize Results (Add --vis to visualize")
    argument_parser_object.add_argument('-t', '--train_json', type=str, default="PennFudanPed_train.json", help="path for the train json file.")
    argument_parser_object.add_argument('-val', '--val_json', type=str, default="PennFudanPed_val.json", help="path for the validation json file.")
    args = argument_parser_object.parse_args()
    main(args)
