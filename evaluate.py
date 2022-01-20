#!/usr/bin/python3
import argparse
import os
import shutil
import time

import cv2

from urllib.parse import urlparse
 
import tensorflow as tf

import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

import workspace as ws

#np.random.seed(50)

class Evaluator(ws.Workspace):
    CLASS_NAMES = "classfile"
    ANNOTATIONS = "annotations"
    IMAGES = "images"
    DIR_EVAL = "eval"
    DIR_PROC = "proc"
    MODELS_PRE = "pre-trained-models"
    IMG_SIZE = 500
    BORDER_SIZE = 25
    THRESHOLD = 0.5
    COLOR = (38,139,210) # Blue or Cyan(42,161,152)

    XML_TEMPLATE="""<annotation>
    <folder>images</folder>
    <filename>%(filename)s</filename>
    <path>%(path)s</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>%(width)s</width>
        <height>%(height)s</height>
        <depth>%(depth)s</depth>
    </size>
    <segmented>0</segmented>
    %(objects)s
</annotation>
"""
    OBJ_TEMPLATE="""    <object>
        <name>%(label)s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%(xn)s</xmin>
            <ymin>%(yn)s</ymin>
            <xmax>%(xx)s</xmax>
            <ymax>%(yx)s</ymax>
        </bndbox>
    </object>
"""

    def __init__(self, workspaces=None, workspace=None, model=None):
        super().__init__(workspaces=workspaces, workspace=workspace)
        self._load_labels()
        # self._trained
        
        if model is None:
            model = workspace
        model_name = model
        url = urlparse(model)
        if url.scheme in ["http", "https"]:
            model_file = os.path.basename(model)
            model_name = self._model_file[:self._model_file.index(".")]
            get_file(fname=model_file,
                     origin=model,
                     cache_dir=self._trained,
                     extract=True)
        tf.keras.backend.clear_session()
        self._model = tf.saved_model.load(os.path.join(self._workspace, ws.Workspace.TRAINED_FOLDER, model_name, "saved_model"))
                
    def detect(self, img):
    
        imgH, imgW, imgC = img.shape
        
        # Resize image so we can easily see on screen
        # m = max(imgH, imgW)
        # if Evaluator.IMG_SIZE < m:
        #     p = ( m - Evaluator.IMG_SIZE ) / m
        #     newH = round(imgH - imgH * p)
        #     newW = round(imgW - imgW * p)
        #     img = cv2.resize(img.copy(), (newW, newH))
        
        # add border to make the box more visible
        # border = Evaluator.BORDER_SIZE
        # img = cv2.copyMakeBorder(img, top=Evaluator.BORDER_SIZE, bottom=Evaluator.BORDER_SIZE, left=Evaluator.BORDER_SIZE, right=Evaluator.BORDER_SIZE, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        imgH, imgW, img3 = img.shape
        
        tfimg = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        tensor = tf.convert_to_tensor(tfimg, dtype=tf.uint8)
        tensor = tensor[tf.newaxis,...]
        
        detections = self._model(tensor)
        
        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        scores = detections['detection_scores'][0].numpy()
        
        indxs = tf.image.non_max_suppression(boxes, scores, max_output_size=50,
        iou_threshold=Evaluator.THRESHOLD, score_threshold=Evaluator.THRESHOLD)
        
        bboxes = []
        if 0 < len(indxs):
            for i in indxs:
                box = tuple(boxes[i])
                score = round(100.0*scores[i])
                index = classes[i]
                label = self._labels[index-1]
                color = Evaluator.COLOR
                display = "%s: %s%%" % (label.upper(), score)
                
                yn, xn, yx, xx = box
                yn, xn, yx, xx = (yn * imgH, xn * imgW, yx * imgH, xx * imgW)
                yn, xn, yx, xx = int(yn), int(xn), int(yx), int(xx)
                bboxes.append({'yn':yn, 'xn':xn, 'yx':yx, 'xx':xx, 'label':label})
                cv2.rectangle(img, (xn, yn), (xx, yx), color=color, thickness=2)
                cv2.putText(img, display, (xn, yn-10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
        
            
        return ({'height':imgH, 'width':imgW, 'depth':img3}, bboxes)
        
        
    def eval(self):
        files = os.listdir(self._images)
        for file in files:
            split = file.split(".")
            if "jpg" == split[-1]:
                hash = split[0]
                xmlfile = os.path.join(self._images, "%s.xml" % hash)
                if not os.path.exists(xmlfile):
                    imgfile = os.path.join(self._images, file)
                    print("Detecting: %s" % imgfile)
                    image = cv2.imread(imgfile)
                    metadata = self.detect(image)
                
                    if 0 < len(metadata[1]):
                        cv2.imshow("Result", image)
                        key = cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        if ord('a') == key: # 97
                            objs = ""
                            for bbox in metadata[1]:
                                objs += Evaluator.OBJ_TEMPLATE % bbox
                            
                            header = metadata[0]
                            header['filename'] = file
                            header['path'] = imgfile
                            header['objects'] = objs
                            # print(Evaluator.XML_TEMPLATE % header)
                            with open(xmlfile, 'w') as f:
                                f.write(Evaluator.XML_TEMPLATE % header)
                        elif ord('q') == key: # 113
                            break
                        else:
                            print("Not writing XML annotation for %s [key=%s]" % (imgfile, key))
                    else:
                        print("No bounding box found for %s" % imgfile)
                    
                    
        else:
            print("No input files found - sleeping(10s)")
        
                

# Based on YouTube video: https://youtu.be/2yQqg_mXuPQ
if "__main__" == __name__:
    parser = argparse.ArgumentParser(description='TF model tester by processing a folder of images for evaluation by adding a bounding box annotation file.')
    parser.add_argument('--workspaces', default=None, help='path to workspaces folder (default: ~/workspaces')
    parser.add_argument('--workspace', help='name of the workspace to process')
    parser.add_argument('--model', default=None, help='url to pre-existing model package')
    args = parser.parse_args()
    
    evaluator = Evaluator(workspaces=args.workspaces, workspace=args.workspace, model=args.model)
    evaluator.eval()
    
