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
from sklearn.svm import SVC
import torch,torchvision
import torch.utils.data 
from utils import *
from torchvision import transforms as T
from PIL import Image

def faster_rcnn(args):
    threshold = 0.75

    if not os.path.exists(str(args.inp_folder)+'/saved_models'):
        os.mkdir(str(args.inp_folder)+'/saved_models')
    if not os.path.exists(str(args.inp_folder)+'/saved_models/faster_rcnn.pth'):
    # Download a pretrained Resnet50 Faster R-CNN model with pretrained weights.
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,pretrained_backbone=True,progress=True)
        # From Pytorch official site -> The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each image, and should be in 0-1 range
        torch.save(model,str(args.inp_folder)+'/saved_models/faster_rcnn.pth')
    else:
        print(f"Found model at {args.inp_folder}/saved_models/faster_rcnn.pth")
        model = torch.load(str(args.inp_folder)+'/saved_models/faster_rcnn.pth')


    if os.path.exists(str(args.inp_folder)+"/vis_faster_rcnn"):
        shutil.rmtree(str(args.inp_folder)+"/vis_faster_rcnn")
    os.mkdir(str(args.inp_folder)+"/vis_faster_rcnn")


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    
    coco_result = []
    category_id = 1

    class PennFudanDataset(torch.utils.data.Dataset):
        def __init__(self, root, val_json_data):
            super(PennFudanDataset, self).__init__()
            self.root = root 

            images_dict=val_json_data['images'] 
            self.image_filenames=[images_dict[i]['file_name'].split('/')[2] for i in range(len(images_dict))]
            
            annotations_dict=val_json_data['annotations'] 
            image_ids=[annotations_dict[i]['image_id'] for i in range(len(annotations_dict))]
            self.image_ids=np.unique(image_ids)

        def __getitem__(self,idx):
            image_path = os.path.join(self.root, "PNGImages",self.image_filenames[idx])
            image_id = self.image_ids[idx]
            # image= cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = image / 255.0
            # image = torch.FloatTensor(image)
            image = Image.open(image_path).convert("RGB")
            image = np.array(image) 
            image=T.ToTensor()(image)
            return image,image_path,image_id

        def __len__(self):
            return len(self.image_ids)

    
    f=open("PennFudanPed_val.json",'r') 
    data=json.load(f)
    dataset_test = PennFudanDataset(args.inp_folder, data)  
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=None) 

    
    with torch.no_grad():
        # Place the model in evaluation mode
        model.to(device)
        model.eval()

        for image, image_path, image_id in tqdm(data_loader_test):
            image = image.to(device)
            original_image=cv2.imread(image_path[0])
            preds = model(image)[0] 
            keep = torchvision.ops.nms(preds['boxes'], preds['scores'], 0.1)
            for ind in keep: 
                ind = ind.item() 
                # print(ind)
                confidence = preds['scores'][ind] 
                # print(confidence.item())  
                if confidence.item() > threshold: ## thresholding 
                    # print('*********')
                    idx = int(preds["labels"][ind])  ###
                    if idx==1:
                        bbox = preds["boxes"][ind].detach().cpu().numpy()  ## for saving boxes  
                        bbox = [int(i) for i in bbox]
                        (startX, startY, endX, endY) = bbox 
                        bbox = [startX, startY, endX-startX, endY-startY]
                        coco_result.append({"image_id":int(image_id),"category_id":int(category_id),"bbox":bbox,"score":np.round(confidence.item(),2)})
                    if args.vis:
                        cv2.rectangle(original_image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                        # cv2.putText(original_image_1 , str(round(score.item(),3)), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if args.vis:
                    # cv2.imshow("Final Detection", original_image)
                    cv2.imwrite(str(args.inp_folder)+"/vis_faster_rcnn/"+str(image_path[0].split('/')[2]),original_image)
   
   
    print(f"Saved predictions at {args.inp_folder+'/pred_eval_faster_rcnn.json'}")
    json.dump(coco_result, open(args.inp_folder+"/pred_eval_faster_rcnn.json", 'w'), ensure_ascii=False) #TODO 
        



def main(args):
    faster_rcnn(args)

if __name__ == "__main__":
    argument_parser_object = argparse.ArgumentParser(description="Pedestrian Detection in images")
    argument_parser_object.add_argument('-i', '--inp_folder', type=str, default='PennFudanPed', help="Path for the root folder of dataset containing images, annotations etc.)")
    argument_parser_object.add_argument('-v', '--vis', action='store_true', default=False, help="Visualize Results (Add --vis to visualize")
    argument_parser_object.add_argument('-t', '--true_json', type=str, default="PennFudanPed_full.json", help="path for the true annotation json file.")
    args = argument_parser_object.parse_args()
    main(args)
