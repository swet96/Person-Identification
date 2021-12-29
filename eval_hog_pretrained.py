""" 
COL780
Assignment 3
"""

import cv2
import argparse
import os, shutil
import json
import numpy as np
from tqdm import tqdm
from utils import *


# Using OpenCV Detector
def hog_pretrained(args):
    # Initialize the HOG detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    if os.path.exists(str(args.inp_folder)+"/vis_hog_pretrained"):
        shutil.rmtree(str(args.inp_folder)+"/vis_hog_pretrained")
    os.mkdir(str(args.inp_folder)+"/vis_hog_pretrained")

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
        image = cv2.imread(os.path.join(args.inp_folder,"PNGImages",file))
        
        h, w = image.shape[:2]

        original_image = image.copy()

        # Running detector
        (pred, confidence) = hog.detectMultiScale(image, winStride=(2, 2), padding=(4, 4), scale=1.05)

        # The size of the sliding window = (64, 128) defualt & as suggested in original paper

        rects = []
        for rect in pred:
            x,y,w,h = rect
            x1 = x
            y1 = y
            x3 = x + w
            y3 = y + h
            rects.append([x1,y1,x3,y3])
        rects=np.array(rects)
        
        rects,scores = NMS(rects,confidence)
        
        for rect,score in zip(rects,scores):
            x1,y1,x3,y3 = rect.tolist()
            coco_result.append({"image_id":int(image_id),"category_id":int(category_id),"bbox":[float(x1),float(y1),float(x3-x1),float(y3-y1)],"score":np.round(score.item(),3)})
            if args.vis:
                cv2.rectangle(original_image, (x1, y1), (x3, y3), (0, 0, 255), 2)
                # cv2.putText(original_image_1 , str(round(score.item(),3)), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if args.vis:
            cv2.imwrite(str(args.inp_folder)+"/vis_hog_pretrained/"+str(file),original_image)
    
    if args.vis:
        print(f"Saved images with bounding box in {args.inp_folder+'/vis_hog_pretrained.json'}")

    print(f"Saved predictions at {args.inp_folder+'/pred_eval_hog_pretrained.json'}")
    json.dump(coco_result, open(args.inp_folder+"/pred_eval_hog_pretrained.json", 'w'), ensure_ascii=False)
        




def main(args):
    hog_pretrained(args)

if __name__ == "__main__":
    argument_parser_object = argparse.ArgumentParser(description="Pedestrian Detection in images")
    argument_parser_object.add_argument('-i', '--inp_folder', type=str, default='PennFudanPed', help="Path for the root folder of dataset containing images, annotations etc.)")
    argument_parser_object.add_argument('-v', '--vis', action='store_true', default=False, help="Visualize Results (Add --vis to visualize")
    argument_parser_object.add_argument('-t', '--train_json', type=str, default="PennFudanPed_train.json", help="path for the train annotation json file.")
    argument_parser_object.add_argument('-val', '--val_json', type=str, default="PennFudanPed_val.json", help="path for the validation annotation json file.")

    args = argument_parser_object.parse_args()
    main(args)