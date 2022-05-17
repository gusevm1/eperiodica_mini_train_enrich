#!/usr/bin/env python
# coding: utf-8

# In[11]:


# This code takes all the images from the eperiodica minitrain and enriches the data by creating a new jason file with
# the ocr text in it. It works as follows:
# 1. It loads the images scenegraph objects (all bounding boxes from the ground truth), scenegraph relationships and file info
# 2. The maximum id for objects and relationships is calculated.
# 3. For each image the ocr text is loaded and then each word of the ocr text is processed.(ocr dimensions must be adjusted)
# 4. A list of all initial bounding boxes of the image is created.
# 5. Loop over all bounding boxes and calculate IOU (intersection over union) with each word of the ocr text
# 6. Create "parent_of" relationship between the best fitting bounding box and word.
# 7. For bounding box create a list of children of the bounding box (each word should only be child of one bounding box)
# 8. Create a followed by relationship between words in the same bounding box
import json
from PIL import Image
import cv2
import torch
import torchvision.ops.boxes as bops
from shapely.geometry import Polygon
import copy


# In[12]:


# define scale function based on ocr and scenegraph dimensions for x coordinates and y coordinates
# this function is needed since the dimensions at docparser/fulltext are not the same dimension as in the the ground truth
def scale_ocr_x(x, dimensions_scenegraph, dimensions_ocr):
    return x * dimensions_scenegraph[0] / dimensions_ocr[0]
def scale_ocr_y(y, dimensions_scenegraph, dimensions_ocr):
    return y * dimensions_scenegraph[1] / dimensions_ocr[1]


# In[13]:


# two seperate implementations of iou

#converts from format x, y, w, h to x1, y1 , x2 , y2
def iou1(bbox1, bbox2): 
    # x1, y1 , x2, y2 format
    bbox1_a = [bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]]
    bbox2_a = [bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]]
    
    # determine the coordinates of the intersection rectangle
    x_left = max(bbox1_a[0], bbox2_a[0])
    y_top = max(bbox1_a[1], bbox2_a[1])
    x_right = min(bbox1_a[2], bbox2_a[2])
    y_bottom = min(bbox1_a[3], bbox2_a[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)
    
    bb1_area = (bbox1_a[2] - bbox1_a[0] + 1) * (bbox1_a[3] - bbox1_a[1] + 1)
    bb2_area = (bbox2_a[2] - bbox2_a[0] + 1) * (bbox2_a[3] - bbox2_a[1] + 1)
    
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


#converts from x, y, w, h format to (x, y) coordinates of all corner points starting top left clockwwise
#then uses the shapely polygon function to return the union and intersection
def iou2(bbox1, bbox2):
    bbox1_a = [[bbox1[0], bbox1[1]], [bbox1[0] + bbox1[2], bbox1[1]], [bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]], [bbox1[0], bbox1[1] + bbox1[3]]]
    bbox2_a = [[bbox2[0], bbox2[1]], [bbox2[0] + bbox2[2], bbox2[1]], [bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]], [bbox2[0], bbox2[1] + bbox2[3]]]
    poly_1 = Polygon(bbox1_a)
    poly_2 = Polygon(bbox2_a)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou


# In[14]:


#returns parent with best overlap with child, parents -> [[x,y,w,h], [x,y,w,h], ..] child -> [x,y,w,h]
#uses iou2, retur
def bounding_box_best_fit(bounding_box_parents, bounding_box_child):
    iou_list = []
    for bbox in bounding_box_parents:
        iou_list.append(iou2(bounding_box_child, bbox))
    
    list_of_maximal_iou_bboxes = []
    for i in range(len(iou_list)):
        if iou_list[i] == max(iou_list):
            list_of_maximal_iou_bboxes.append(i)
    return list_of_maximal_iou_boxes


# In[15]:


# load objects, relationships and image information
f_iminfo = open('/mnt/ds3lab-scratch/gusevm/data/eperiodica2/anns/train/eperiodica_minitrain/eperiodica_minitrain_VG_scene_graph_image_data.json')
iminfo = json.load(f_iminfo)

f_obj = open('/mnt/ds3lab-scratch/gusevm/data/eperiodica2/anns/train/eperiodica_minitrain/eperiodica_minitrain_VG_scene_graph_objects.json')
obj = json.load(f_obj)

f_rel = open('/mnt/ds3lab-scratch/gusevm/data/eperiodica2/anns/train/eperiodica_minitrain/eperiodica_minitrain_VG_scene_graph_relationships.json')
rel = json.load(f_rel)
###


# In[16]:


# open the destination json files
f_obj_enriched = open('/mnt/ds3lab-scratch/gusevm/eperiodicaOCR/eperiodica_minitrain_VG_scene_graph_objects_enriched.json')
obj_enriched = json.load(f_obj_enriched)

f_rel_enriched = open('/mnt/ds3lab-scratch/gusevm/eperiodicaOCR/eperiodica_minitrain_VG_scene_graph_relationships_enriched.json')
rel_enriched = json.load(f_rel_enriched)


# In[17]:


# copy the contents of the original data
obj_enriched = copy.deepcopy(obj)

rel_enriched = copy.deepcopy(rel)


# In[18]:


#calculates the maximum id of bounding boxes that are totally present, in order to add new bounding boxes and not mess up the indexing
maximum_id_obj = 0
for ob in obj:
    for o in ob['objects']:
        if o['object_id'] > maximum_id_obj:
            maximum_id_obj = o['object_id']
#calculated highest object_id so we add one to start the indexing of the new bounding boxes
maximum_id_obj += 1
print(maximum_id_obj)


#calculates amount of relationships that are totally present, in order to add new relationships without messing up the indexing
maximum_id_relationships = 0
for re in rel:
    for r in re['relationships']:
        if r['relationship_id'] > maximum_id_relationships:
            maximum_id_relationships = r['relationship_id']
#again indexing for new relationships will start one above the current maximum id
maximum_id_relationships += 1
print(maximum_id_relationships)    


# In[19]:


# load image ocr text and combine it with the ground truth

########## 

#counter for test runs to only do 2 -3 images at a time with an if/break
counter = 0

##########

for info in iminfo:
    #loads image by using file name
    image_id = info['image_id']
    path = info['file_name']
    path = path.split(".", 4)
    f = open('/mnt/ds3lab-scratch/docparser/data/eperiodica/fulltext/' + path[0] + "/" + path[2] + "_" + path[3] + "/" +  path[4][5:-6].replace(".", "-", 1).replace(".", "_") + ".txt")
    
    #dimensions of ocr text
    dimensions_ocr = f.readline()
    dimensions_ocr = dimensions_ocr.split(",", 2)
    dimensions_ocr[0] = float(dimensions_ocr[0])
    dimensions_ocr[1] = float(dimensions_ocr[1])
    
    #dimensions of scenegraph annotations
    dimensions_scenegraph = [info['width'], info['height']]
    
    #reads the entire ocr text
    lines = f.readlines()
    
    #creates a list of bounding boxes in current image
    bbox_list = []
    for o in obj[image_id]['objects']:
        bbox = [o['x'], o['y'], o['w'], o['h']]
        bbox_list.append(bbox)
    #remembers the current amount of bboxes in the picture
    amount_of_bbox_current = len(bbox_list)
    print(amount_of_bbox_current)
    
    
    #list that will hold words of bounding boxes, storing object_ids
    bbox_words = []
    for i in bbox_list:
        bbox_words.append([])
    print(bbox_words)
    
    
    #for each line of the ocr text it calculates the intersection over union of the word with all bounding boxes
    for l in lines:
        ## maybe keep information of eop and eos
        #line format is: August x, y, w, h
        if l != "<EOP>\n" and l != "<EOS>\n":
            l = l.split(" ", 1)
            l[1] = l[1].split(",", 4)
        
            x = scale_ocr_x(float(l[1][0]), dimensions_scenegraph, dimensions_ocr)
            y = scale_ocr_y(float(l[1][1]), dimensions_scenegraph, dimensions_ocr)
            w = scale_ocr_x(float(l[1][2]), dimensions_scenegraph, dimensions_ocr)
            h = scale_ocr_y(float(l[1][3]), dimensions_scenegraph, dimensions_ocr)
        
            print(l[0] + ",                     " + str(x) + ",  " + str(y) + ",  " + str(w) + ",  " + str(h))
            
            
            iou_list = []
            xywh_list = [x, y, w, h]
            
            #creates list of intersection over unions
            for bbox in bbox_list:
                iou_list.append(iou2(xywh_list, bbox))
            
            #usually there is only one bounding box with best iou value, but in rare occasions there can be 2, this happens
            #because of a design choice of the ground truth annotations, we deal with this by choosing to include the is_parent_of
            #relationship to both bounding boxes for simplicity, to that end we just create a list of bounding boxes that have
            #maximal iou
            list_of_maximal_iou_bboxes = []
            for i in range(len(iou_list)):
                if iou_list[i] == max(iou_list):
                    list_of_maximal_iou_bboxes.append(i)
            
            #if there's 2 bboxes with maximal iou we assume that one of them must be the parent of the other if there's only one we
            #choose the first element
            #if there are more than 3 bounding boxes with maximal iou we'll have to use some other solution
            maximal_bbox = list_of_maximal_iou_bboxes[0]
            if len(list_of_maximal_iou_bboxes) > 1:
                #if first of the two is the child make it the maximal_bbox
                for r in rel[image_id]['relationships']:
                    if r['predicate'] == "parent_of" and r['object']['object_id'] == list_of_maximal_iou_bboxes[0] and r['subject']['object_id'] == list_of_maximal_iou_bboxes[1]:
                        maximal_bbox = list_of_maximal_iou_bboxes[0]
                        
                #if second of the two is the child make it the maximal_bbox
                for r in rel[image_id]['relationships']:
                    if r['predicate'] == "parent_of" and r['object']['object_id'] == list_of_maximal_iou_bboxes[1] and r['subject']['object_id'] == list_of_maximal_iou_bboxes[0]:
                        maximal_bbox = list_of_maximal_iou_bboxes[1]
            
            #print(str(list_of_maximal_iou_bboxes) + ",  " + str(max(iou_list)) + ",      child is: " + str(maximal_bbox) )
            
            # for each word a new object is created with names "word" and additionally the ocr word is saved along with it in
            # names
            obj_enriched[image_id]['objects'].append({"synsets": [], "x": x, "y": y, "w": w, "h": h, "object_id": maximum_id_obj, "merged_object_ids": [],"names": ["word"], "text": [l[0]]})
            
            #add object_id to bounding box correlated in bbox_words list
            bbox_words[maximal_bbox].append(maximum_id_obj)
            
                
            #append the parent_of relationship to the scene graph relationships
            
            #object of the parent_of relationship
            object_x = obj_enriched[image_id]['objects'][amount_of_bbox_current]['x']
            object_y = obj_enriched[image_id]['objects'][amount_of_bbox_current]['y']
            object_w = obj_enriched[image_id]['objects'][amount_of_bbox_current]['w']
            object_h = obj_enriched[image_id]['objects'][amount_of_bbox_current]['h']
            
            rel_object = {"name": "word", "object_id": maximum_id_obj, "synsets": [], "x": object_x, "y": object_y, "w": object_w, "h": object_h}
            
            #subject of the parent_of relationship
            subject = obj_enriched[image_id]['objects'][maximal_bbox]
            subject_x = subject['x']
            subject_y = subject['y']
            subject_w = subject['w']
            subject_h = subject['h']
                
            rel_subject = {"name": subject['names'][0], "object_id": subject['object_id'], "synsets": [], "x": subject_x, "y": subject_y, "w": subject_w, "h": subject_h}
                
            #append
            rel_enriched[image_id]['relationships'].append({"relationship_id": maximum_id_relationships, "predicate:": "parent_of", "synsets": [], "object": rel_object, "subject": rel_subject})
                
                
            #print(rel_enriched[image_id]['relationships'])
                
            
                 
            #increment amount of bbox's and relationships
            maximum_id_obj += 1
            amount_of_bbox_current += 1
            maximum_id_relationships += 1
    
    
    #create the followed_by relationship for words in the same bounding box
    for l in bbox_words:
        #if list of words for bounding box isn't empty (empty list returns false boolean)
        if not l:
            for words in l[:-1]:
                #object
                o1_x = obj_enriched[image_id]['objects'][words + 1]['x']
                o1_y = obj_enriched[image_id]['objects'][words + 1]['y']
                o1_w = obj_enriched[image_id]['objects'][words + 1]['w']
                o1_h = obj_enriched[image_id]['objects'][words + 1]['h']
            
                rel_object1 = {"name": "word", "object_id": words + 1, "synsets": [], "x": o1_x, "y": o1_y, "w": o1_w, "h": o1_h}
            
                #subject
                s1_x = obj_enriched[image_id]['objects'][words]['x']
                s1_y = obj_enriched[image_id]['objects'][words]['y']
                s1_w = obj_enriched[image_id]['objects'][words]['w']
                s1_h = obj_enriched[image_id]['objects'][words]['h']
            
                rel_subject = {"name": "word", "object_id": words, "synsets": [], "x": s1_x, "y": s1_y, "w": s1_w, "h": s1_h}
            
                rel_enriched[image_id]['relationships'].append({"relationship_id": maximum_id_relationships, "predicate:": "followed_by", "synsets": [], "object": rel_object1, "subject": rel_subject})
                maximum_id_relationships += 1
    
    ##########
    #open image and display all bounding boxes on it
    #check different boxes
    
    im_file = "/mnt/ds3lab-scratch/gusevm/data/eperiodica2/imgs/train_images/" + path[4][5:]
    image = cv2.imread(im_file)
    #
    #
    #
    #
    #
    #bbox_check = 5
    #box_x = int(obj_enriched[image_id]['objects'][bbox_check]['x'])
    #box_y = int(obj_enriched[image_id]['objects'][bbox_check]['y'])
    #box_w = int(obj_enriched[image_id]['objects'][bbox_check]['w'])
    #box_h = int(obj_enriched[image_id]['objects'][bbox_check]['h'])
    #cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 255), 2)
    #
    #for word in bbox_words[bbox_check]:
    #    for o in obj_enriched[image_id]['objects']:
    #        if o['object_id'] == word:
    #            bbox_x = int(o['x'])
    #            bbox_y = int(o['y'])
    #            bbox_w = int(o['w'])
    #            bbox_h = int(o['h'])
    #            cv2.rectangle(image, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (36, 255, 12), 2)
    #cv2.imwrite("/mnt/ds3lab-scratch/gusevm/eperiodicaOCR/temp.png", image)
    #im = Image.open("/mnt/ds3lab-scratch/gusevm/eperiodicaOCR/temp.png")
    #display(im)
    
    
    
    
    for o in obj_enriched[image_id]['objects']:
        bbox_x = int(o['x'])
        bbox_y = int(o['y'])
        bbox_w = int(o['w'])
        bbox_h = int(o['h'])
        if o['names'][0] == "word":
            cv2.rectangle(image, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (36, 255, 12), 2)
        else:
            cv2.rectangle(image, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 0, 255), 2)
    cv2.imwrite("/mnt/ds3lab-scratch/gusevm/eperiodicaOCR/temp.png", image)
    im = Image.open("/mnt/ds3lab-scratch/gusevm/eperiodicaOCR/temp.png")
    display(im)
    
    ##########
    
    
    
    
    
    ##########
    # take few high level things and check that it's actually correct
    print("____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________")
    
    counter += 1
    #if counter == 3:
    #    break
    
    
    print(bbox_words)
    print("____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________")
    
    ##########
    
    
    #print(rel_enriched[image_id]['relationships'])


# In[20]:


#json file indent = 1, keys sorted etc.
with open('/mnt/ds3lab-scratch/gusevm/eperiodicaOCR/eperiodica_minitrain_VG_scene_graph_objects_enriched.json', "w") as file1:
    json.dump(obj_enriched, file1, indent = 2)

with open('/mnt/ds3lab-scratch/gusevm/eperiodicaOCR/eperiodica_minitrain_VG_scene_graph_relationships_enriched.json', "w") as file2:
    json.dump(rel_enriched, file2, indent = 2)


# In[ ]:





# In[ ]:





# In[ ]:




