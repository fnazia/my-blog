---
layout: post
title: "Jaccard Index in Object Detection"
date: 2024-05-12
#permalink: /jaccard-index-in-object-detection/
---

# Jaccard Index in Object Detection

Jaccard Index is defined as the ratio of the size of two sets' intersection and union and is named after Paul Jaccard who introduced the metric as a similarity coefficient in his work on [the distribution of flora in the Alpine zone](https://nph.onlinelibrary.wiley.com/doi/10.1111/j.1469-8137.1912.tb05611.x). It is a similarity index that measures the degree of similarity between two entities. According to the definition, if A and B are two sets, Jaccard Index is -

$$
\begin{align*}
J(A,B) &= \frac{\vert{A}\cap{B}\vert}{\vert{A}\cup{B}\vert} \\
~\\
\end{align*}
$$

This Index is used in Object Detection problems of Machine Learning and is generally referred to as IoU or Intersection-over-Union in the field of Computer Vision. Object detection problem not only needs to recognize the objects or events in an image or video, it also has to propose the region of these detections. Algorithms devised to solve this problem, such as YOLO, SSD, Faster R-CNN, predict the location of the objects, which is also known as Region-of-Interest (RoI), by using [Non-Maximum Suppression (NMS)](https://en.wikipedia.org/wiki/Edge_detection#Canny) <sup>[2](http://kth.diva-portal.org/smash/record.jsf?pid=diva2%3A452310&dswid=9254) </sup> method. This NMS method utilizes the metric [Jaccard Index](https://en.wikipedia.org/wiki/Jaccard_index) to determine which locations, among hundreds of those predicted by the detection model, should remain in the final prediction. This metric is used in the evaluation of the performance of the model as well.

Usually Jaccard Overlap is used to measure the similarity or overlap between the predicted and the actual locations of the detected objects. The location and size of an object is defined by a Bounding Box that helps to localize the object in an image. Bounding boxes in the 2D images are usually represented by the x-, y-coordinate of the object's center, and height and width of a rectangular box surrounding the target object. IoU measures the overlap between two such bounding boxes in the use case of Object Detection in 2D images.

Let us take a look at how IoU is used in the object detection models. Here Jaccard Overlap is used in three different parts of the problem -

1. Compute the loss</br>
2. Predict the objects' location and filter out the overlapping (similar) predictions in the unseen samples</br>
3. Calculate mean Average Precision (mAP)

This [series](https://numbersmithy.com/create-yolov3-using-pytorch-from-scratch-part-1/) explains an object detection algorithm ([YOLO](https://arxiv.org/abs/1506.02640)) in details. We will briefly touch upon only those concepts that are required to measure the IoU. The input images passed into the model are usually in the shape of \[N, C, H, W\], where N is the number of images, C is the number of channels of each image, and H and W are respectively the height and the width of the image. 

{::options parse_block_html="true" /}

<details><summary markdown="span"><code>Expand Code</code></summary>

```python
# Import packages

import matplotlib.pyplot as plt
import numpy as np
import torch
```


```python
# Illustration of objects inside bounding boxes

from matplotlib.patches import Circle, Ellipse, Polygon

# Define circle params

center = (3, 4) # (0.3, 0.4) 
radius = 2 # 0.1
circle = Circle(center, radius, fill = False)

# Define ellipse params

ellipse_xy = (9, 8) #(0.7, 0.7)
width = 6 # 0.4
height = 3 # 0.2
angle = 35
ellipse = Ellipse(xy=ellipse_xy, width=width, height=height, angle=angle, fill = False, edgecolor="k") #, facecolor = 'none', 

figure, axes = plt.subplots()

# Set plot axes limit according to stride

axes.set_xlim([0, 13])
axes.set_ylim([0, 13])

axes.set_aspect(1)

# Draw circle

axes.add_artist(circle)

# Draw ellipse

axes.add_artist(ellipse)

# Define circle bbox and draw

circle_polygon = np.array([[0.8*(center[0] - radius), 0.9*(center[1] - radius)], 
                           [1.05*(center[0] + radius), 0.9*(center[1] - radius)], 
                           [1.05*(center[0] + radius), 1.05*(center[1] + radius)], 
                           [0.8*(center[0] - radius), 1.05*(center[1] + radius)]])

circle_bbox = Polygon(circle_polygon, closed=True, fill = False, edgecolor='r')

axes.add_artist(circle_bbox)

# Define ellipse bbox and draw

ellipse_polygon = np.array([[2.05*(ellipse_xy[0] - width), 1.1*(ellipse_xy[1] - height)], 
                            [0.79*(ellipse_xy[0] + width), 1.1*(ellipse_xy[1] - height)], 
                            [0.79*(ellipse_xy[0] + width), 0.95*(ellipse_xy[1] + height)], 
                            [2.05*(ellipse_xy[0] - width), 0.95*(ellipse_xy[1] + height)]])

ellipse_bbox = Polygon(ellipse_polygon, closed=True, fill = False, edgecolor='r')

axes.add_artist(ellipse_bbox)



# Draw gridlines 

plt.xticks(list(np.arange(14)))
plt.yticks(list(np.arange(14)))
plt.grid(color = 'g')

plt.show()
```
</details>

{::options parse_block_html="false" /}

    
![png](/assets/images/Jaccard_object_detection_files/Jaccard_object_detection_7_0.png)
    


The detection models usually consider an image to be divided into certain number of grid cells and aim to generate feature maps similar to the size of this grid size. For example, with a stride of 32 an image of shape or pixels (416x416) can be reduced down to a feature map of shape or grid cells (13x13). An illustration of this is shown in the image above, where the two objects are the circle and the ellipse and the red rectangular boxes surrounding them are the bounding boxes. Feature maps of such specific size is extracted for the objects of a particular scale. Larger object's detection requires feature maps of smaller number of grid cells (13x13) and smaller object's detection requires feature maps of larger number of grid cells (52x52). That is why, predictions are made in multiple scales to generate feature maps of different sizes. For each of the grid cells in the feature map a matrix of size \[n_anchors, 4+1+n_classes\] is predicted, where n_anchors is the number of anchor boxes, 4 is for the location specifications (x-coord, y-coord, height, width) of the predicted bounding box, 1 is for the probability score (or IoU score) of an object being present in the grid cell (called objectness score), and n_classes is the number of probability scores representing the object being one of the total number of n classes. 

Anchor box is a concept unique to Object Detection models like YOLO. These boxes are predefined rectangular boxes of certain sizes corresponding to the sizes of the objects in the training dataset and are used to predict localization. In some cases, anchor boxes are determined based on k-means clustering of the actual bounding boxes in the training dataset; or sometimes these are calculated from the image size using fixed scales. Therefore, the total number of predicted boxes or locations for a single image is (n_scales x n_anchors x n_Ycells x n_Xcells). From these predictions the overlapping and irrelevent ones are eliminated with the help of Jaccard Index or IoU.

When inferring the location of the objects in the unseen samples, the bounding boxes (x-center offset, y-center offset, height, and width) are predicted in fractional values of a single grid cell. These fractional values of the predicted bounding boxes are rescaled to the image's original pixel values - the height and width are adjusted corresponding to their respective anchor boxes and the x, y offsets are added to their corresponding grid cell's starting point to find their location in the feature maps before transforming them into pixel values using the corresponding stride value. Only those boxes having objectness score higher than a threshold are compared with each other in terms IoU. If the IoU score is greater than a certain threshold for a pair of predicted boxes, one of them is dropped from the final prediction. This is how the overlapping predicted boxes are filtered out. 

When computing the loss of an object's location in the image, the object is mapped to the anchor box most similar to itself. Although (n_scales x n_anchors x n_Ycells x n_Xcells) number of boxes are predicted for each image regardless of the number of objects contained by it, the bounding box loss is computed between only one of these predicted boxes and the ground-truth box of an object. The anchor box positioned at the grid-cell containing the center of the target object and matching the object's size most closely is calculated to compare with the ground-truth location. This comparison is performed using IoU between the two boxes and (1-IoU) is minimized to reach the convergence of the model.

Finally, when evaluating the model, the mean Average Precision (mAP) metric utilizes IoU to measure the similarity between the actual bounding box and the predicted bonding box. An object's actual bounding box is compared with all the predicted boxes proposed for it and the one with the highest IoU is considered to be the final prediction.

## Demonstration

Let us assume that an image containing two objects is passed into an object detection model. The actual bounding boxes of the two objects and the output corresponding to (13x13) feature map for three predefined anchor boxes are represented in the following manner. All the values are in the fractional scale (between 0 and 1).

{::options parse_block_html="true" /}

<details><summary markdown="span"><code>Expand Code</code></summary>

```python
# Define actual bounding box input and predicted bounding box output

bbox_actual = np.random.uniform(0, 1, (1, 2, 4)) # shape [n_batch, n_labels coords]
bbox_pred = np.random.uniform(0, 1, (1, 3, 13, 13, 4)) # shape [n_batch, n_anchors, grid_size_x, grid_size_y, coords]

print(f'bbox_actual shape: {bbox_actual.shape}\n\
bbox_pred shape: {bbox_pred.shape}')
```

    bbox_actual shape: (1, 2, 4)
    bbox_pred shape: (1, 3, 13, 13, 4)

</details>

{::options parse_block_html="false" /}

{::options parse_block_html="true" /}

<details><summary markdown="span"><code>Expand Code</code></summary>

```python
# actual bbox

print(f'bbox_actual:\n \
shape: {bbox_actual.shape}\n \
array:\n \
{bbox_actual}')
```
</details>

{::options parse_block_html="false" /}

    bbox_actual:
     shape: (1, 2, 4)
     array:
     [[[0.21290161 0.29416328 0.70791082 0.83901294]
      [0.85741041 0.78432817 0.96968387 0.33843467]]]

{::options parse_block_html="true" /}

<details><summary markdown="span"><code>Expand Code</code></summary>

```python
# predicted bbox

print(f'bbox_pred:\n \
shape: {bbox_pred.shape}\n \
array:\n \
{bbox_pred}')
```

    bbox_pred:
     shape: (1, 3, 13, 13, 4)
     array:
     [[[[[2.90202404e-01 2.97033708e-01 8.09306653e-01 9.46463058e-01]
        [6.21716045e-01 5.12224689e-01 5.29561432e-01 4.86934504e-01]
        [6.97149485e-01 6.28725020e-01 9.65425586e-01 8.16918888e-01]
        ...
        [2.97465090e-01 8.84212252e-01 2.65157926e-01 9.36648609e-01]
        [2.54653850e-01 3.56487368e-01 1.60115920e-01 6.22211707e-01]
        [6.70351290e-01 2.01811087e-01 1.03300127e-01 3.91844665e-01]]
    
       [[6.41831998e-01 4.73080836e-01 6.37786226e-01 4.38285722e-02]
        [4.94995960e-01 3.86608172e-01 1.57460834e-01 5.86177339e-02]
        [1.44528892e-02 5.95272174e-01 1.14278568e-01 5.62223401e-01]
        ...
        [2.40180949e-01 5.99076406e-02 6.36826113e-01 7.97736311e-01]
        [4.74768110e-01 6.99419849e-01 6.11762340e-01 1.28014177e-01]
        [5.08870924e-01 6.05657504e-01 3.27296021e-02 7.89980665e-01]]
    
       [[5.69143326e-01 7.71754097e-02 5.60440137e-01 4.56682754e-01]
        [8.60543647e-03 4.55763943e-01 2.96235127e-01 5.19662313e-01]
        [2.86897244e-01 3.78413043e-01 1.04338408e-01 5.67445243e-02]
        ...
        [7.26835227e-02 5.18918938e-01 5.18799548e-01 2.39952753e-01]
        [1.42863360e-01 1.29798241e-01 7.59553391e-01 5.89253951e-02]
        [5.87811388e-01 9.14117207e-01 5.09926163e-01 5.83944289e-01]]
    
       ...
    
       [[7.01422873e-01 3.46631930e-01 2.14789547e-01 2.55101760e-01]
        [8.14399518e-01 6.07796060e-01 9.33915509e-01 4.75909105e-01]
        [5.68730144e-01 4.27340562e-01 7.98949705e-01 9.80432164e-01]
        ...
        [7.51454024e-01 4.45154063e-01 1.78291568e-01 3.52390894e-01]
        [6.17631499e-01 9.69119765e-01 9.06422185e-01 1.95940791e-01]
        [8.31168057e-01 7.63280398e-01 2.11636028e-01 3.48273935e-01]]
    
       [[7.64191836e-01 4.33642929e-01 7.34330415e-01 4.95755274e-01]
        [7.60540665e-01 9.35857211e-01 4.96625484e-01 3.47074903e-01]
        [4.67811375e-01 4.77164118e-01 2.58212749e-01 8.93719584e-01]
        ...
        [3.59451810e-01 5.39897745e-01 3.53011692e-01 7.39518341e-01]
        [8.96261857e-01 7.85701120e-03 5.58106247e-01 4.65064695e-01]
        [5.64564939e-01 1.48067163e-01 7.45984178e-01 6.22594059e-01]]
    
       [[1.76800252e-01 2.96700244e-02 8.72336327e-01 8.34534797e-01]
        [8.73754425e-01 6.97593265e-03 6.49634982e-01 6.19126172e-01]
        [5.28831409e-01 4.11462519e-01 3.33093462e-01 9.03558193e-01]
        ...
        [9.54565119e-01 9.65157531e-02 5.07095382e-01 6.39050744e-01]
        [1.18904406e-01 1.73722908e-01 2.89492730e-01 3.84553326e-01]
        [5.29056315e-02 1.04798520e-01 9.39252733e-02 3.63231885e-01]]]
    
    
      [[[4.10058954e-01 7.93634832e-01 1.37385893e-01 2.25409924e-01]
        [5.98622145e-01 5.91248128e-01 4.72513839e-01 7.64264888e-01]
        [6.21431187e-01 6.46368682e-01 5.19466876e-01 3.88683677e-01]
        ...
        [9.66354507e-01 4.18669252e-01 6.65248961e-01 7.13981564e-01]
        [6.60570962e-01 4.99842518e-01 7.14376436e-01 8.91206947e-01]
        [4.67515627e-01 4.39973411e-01 5.65246173e-01 7.05949477e-01]]
    
       [[8.95207068e-01 2.78639160e-01 5.52379607e-01 1.58201518e-01]
        [2.02260117e-01 8.53515748e-01 2.60613420e-01 8.10384408e-01]
        [4.57047554e-02 1.53247955e-01 5.77090027e-01 8.66740249e-01]
        ...
        [5.48891609e-01 2.68916782e-01 7.40493187e-01 8.11874983e-01]
        [8.53296702e-01 7.96966991e-01 3.69843866e-01 4.75261085e-01]
        [4.09918306e-01 1.98545933e-01 9.46313501e-01 7.57606802e-01]]
    
       [[8.03188070e-01 5.80602211e-01 5.72611816e-01 9.22428293e-02]
        [3.86019485e-01 5.02013706e-01 2.16020901e-01 6.46903563e-01]
        [9.23582046e-02 3.35710529e-01 9.87248718e-01 8.67787601e-01]
        ...
        [9.27764886e-01 5.58686586e-01 7.98783101e-02 1.83061726e-01]
        [5.28981275e-01 6.32834860e-01 4.64356137e-02 9.72947053e-01]
        [6.45774793e-01 5.43227614e-01 5.07242056e-04 2.36880229e-01]]
    
       ...
    
       [[4.95476044e-01 3.74247250e-02 8.25763393e-01 6.68082755e-01]
        [4.28317275e-01 1.49115949e-02 2.97694881e-01 8.96254295e-01]
        [9.38694486e-01 8.92086639e-01 5.10960630e-01 9.34969051e-01]
        ...
        [4.37650156e-01 6.79340258e-01 6.86170839e-01 7.13485933e-02]
        [9.15362641e-02 7.80738538e-01 5.93383534e-02 3.53080630e-01]
        [3.44465733e-01 9.88582916e-01 1.24654506e-02 6.52470591e-01]]
    
       [[9.90846744e-01 8.34338194e-01 4.76075527e-01 5.02647203e-02]
        [6.24786795e-01 3.53401950e-01 2.19452373e-02 2.74030151e-01]
        [4.09451282e-01 9.89894266e-01 9.47025721e-01 7.42625687e-01]
        ...
        [5.83262293e-02 3.20281634e-01 2.35246999e-01 8.26216490e-01]
        [6.17621291e-01 8.99995777e-01 9.10103309e-01 2.24715989e-01]
        [8.81884907e-01 5.48318450e-01 2.73154947e-01 9.14220475e-01]]
    
       [[4.32109254e-01 4.98552035e-01 3.56770430e-01 7.49166353e-01]
        [9.65837205e-01 5.91638995e-01 8.95881400e-01 5.06528228e-01]
        [7.41322157e-01 8.15358048e-01 1.12641526e-01 3.33486360e-02]
        ...
        [6.94197665e-01 1.06773888e-01 8.63567519e-01 5.21358801e-01]
        [2.88813561e-01 4.82869649e-01 3.78066622e-01 9.84058010e-01]
        [9.69438507e-01 4.01917451e-01 1.24480892e-01 7.13058015e-01]]]
    
    
      [[[9.02825925e-01 4.46724959e-01 6.04128056e-01 7.94624978e-01]
        [6.69689583e-01 7.77161924e-01 2.97462924e-01 2.92555910e-01]
        [4.30310291e-01 2.85698428e-01 6.61926946e-01 3.36843212e-01]
        ...
        [4.86306856e-01 7.28593013e-01 4.72300051e-01 8.55682967e-01]
        [3.52385849e-01 5.25778798e-01 7.92492497e-01 4.51796574e-01]
        [5.70186755e-02 3.86377477e-01 7.09432313e-01 4.17376611e-01]]
    
       [[6.43554262e-02 2.91250539e-01 3.99838764e-01 5.44581175e-01]
        [8.64191695e-01 1.68348312e-01 7.74921180e-01 5.61380683e-01]
        [2.13004514e-01 2.67024110e-01 7.20191693e-01 2.89069493e-01]
        ...
        [2.51428536e-01 5.81320386e-01 2.75156175e-01 1.95744903e-01]
        [4.52652310e-01 6.53335836e-01 9.05311514e-01 9.56486911e-01]
        [1.57996364e-01 5.06224333e-01 9.62607366e-01 3.33886899e-01]]
    
       [[5.74666457e-02 2.31199831e-01 1.38586297e-01 6.31746523e-01]
        [8.95938013e-01 8.22407889e-01 3.38675487e-01 1.72736223e-01]
        [8.47404718e-01 3.91186630e-01 8.09560122e-01 6.88834606e-01]
        ...
        [9.33664243e-01 9.31173741e-01 7.45591098e-01 6.10370010e-01]
        [5.40006992e-01 2.98732978e-01 5.89333135e-01 7.20250601e-01]
        [4.92211334e-01 5.95317017e-01 7.44683642e-01 2.89413749e-01]]
    
       ...
    
       [[7.23627221e-01 8.02106394e-01 4.14460596e-01 6.93875307e-01]
        [2.37738500e-01 5.78356689e-01 1.26739249e-01 7.95486311e-01]
        [6.64800929e-01 9.10885202e-01 6.00202358e-01 2.79244678e-01]
        ...
        [4.82663147e-01 1.12177596e-01 6.72561199e-03 8.35959884e-01]
        [4.00655130e-01 1.36103402e-01 5.81611498e-01 6.68203825e-01]
        [5.50428782e-02 1.53477203e-01 3.57446424e-01 6.63959586e-01]]
    
       [[6.44445656e-01 7.86210591e-01 6.18832137e-01 7.12579892e-01]
        [7.63769566e-01 5.47208806e-01 5.19628102e-01 1.45520194e-01]
        [6.97759397e-01 4.60371770e-01 1.55126390e-01 3.97695080e-01]
        ...
        [7.75208405e-01 5.23817743e-01 5.05925496e-01 9.36259808e-01]
        [3.90098246e-01 2.91350489e-01 7.64383915e-01 9.88982289e-01]
        [1.58753409e-02 9.02066829e-01 8.99679638e-02 7.80315310e-01]]
    
       [[2.08548554e-02 8.05778619e-01 9.56809948e-01 5.39017049e-02]
        [2.67960505e-02 3.93104771e-01 7.10979583e-01 4.90517977e-01]
        [1.19806440e-01 3.09438645e-01 4.90369952e-01 6.08409690e-01]
        ...
        [1.75030044e-01 4.81203048e-01 9.43234238e-01 1.97801543e-03]
        [9.02699152e-01 5.87431140e-01 2.27004269e-01 3.70325974e-01]
        [3.36058041e-02 2.21925928e-01 8.36601127e-01 9.75387326e-01]]]]]

</details>

{::options parse_block_html="false" /}

Converting the fractional scale to feature scale (values should be between 0 and 12) for grid cell position (3, 3) and for first and third anchor boxes we get the following form of predicted bounding box. Calculating the IoU of these two pairs of actual and predicted boxes we see that the first predicted box overlaps with the first actual box almost 42\%; but the second predicted box does not overlap with the second actual box. Actual bounding boxes' scale is changed to feature scale as well. 

{::options parse_block_html="true" /}

<details><summary markdown="span"><code>Expand Code</code></summary>

```python
# actual and predicted bbox in grid scale

bbox_actual_grid_scale = bbox_actual[0]*13
bbox_pred_grid_scale = bbox_pred[0, [0, 2], 3, 3, :]*13

print(f'bbox_actual_grid_scale:\n \
{bbox_actual_grid_scale} \n\
bbox_pred_grid_scale:\n \
{bbox_pred_grid_scale}')
```
</details>

{::options parse_block_html="false" /}

    bbox_actual_grid_scale:
     [[ 2.76772099  3.82412258  9.20284061 10.90716819]
     [11.14633535 10.19626615 12.60589032  4.39965071]] 
    bbox_pred_grid_scale:
     [[ 6.27252577  6.24175572 11.23818034  8.57538178]
     [12.15843153  3.54273941  9.59581098  0.71452057]]

{::options parse_block_html="true" /}

<details><summary markdown="span"><code>Expand Code</code></summary>

```python
# functions definitions to find box's coordinates and to calculate jaccard overlap

def box_coords(box):
    x1, y1, x2, y2 = box[:, 0:1] - (box[:, 2:3]/2), box[:, 1:2] - (box[:, 3:4]/2), \
    box[:, 0:1] + (box[:, 2:3]/2), box[:, 1:2] + (box[:, 3:4]/2)
    
    x1, y1, x2, y2 = np.clip(x1, 0, 12), np.clip(y1, 0, 12), np.clip(x2, 0, 12), np.clip(y2, 0, 12)
    return x1, y1, x2, y2
    
def calc_jaccard_overlap(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box_coords(box1)
    box2_x1, box2_y1, box2_x2, box2_y2 = box_coords(box2)
    
    left_intersect_x1 = np.maximum(box1_x1, box2_x1)
    right_intersect_x2 = np.minimum(box1_x2, box2_x2)
    top_intersect_y1 = np.maximum(box1_y1, box2_y1)
    bottom_intersect_y2 = np.minimum(box1_y2, box2_y2)
    
    width_intersect = np.clip(right_intersect_x2-left_intersect_x1, 0, None)
    height_intersect = np.clip(bottom_intersect_y2-top_intersect_y1, 0, None)
    
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    
    intersection_area = width_intersect*height_intersect
    union_area = box1_area+box2_area-intersection_area
    
    iou = intersection_area/union_area
    
    return iou
```
</details>

{::options parse_block_html="false" /}

Here are the IoUs of the two pairs of actual and predicted boxes -

{::options parse_block_html="true" /}

<details><summary markdown="span"><code>Expand Code</code></summary>

```python
# Check if an acceptable pair of bboxes are generated for the demonstration

calc_jaccard_overlap(bbox_actual_grid_scale, bbox_pred_grid_scale)
```

</details>

{::options parse_block_html="false" /}


    array([[0.42562048],
           [0.        ]])



Here's the plot of the overlapping pair of boxes. The green box is the actual bounding box and the blue box is the predicted bounding box. The area with red hatchet lines is the overlapped area between two boxes. 

{::options parse_block_html="true" /}

<details><summary markdown="span"><code>Expand Code</code></summary>

```python
# Draw the overlap between two bboxes

i = 0
j = 0
x1, y1, x2, y2 = np.hstack(box_coords(bbox_actual_grid_scale))[i]
x1p, y1p, x2p, y2p = np.hstack(box_coords(bbox_pred_grid_scale))[j]

plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color='g', linewidth=3.0, label='actual')
plt.plot([x1p, x2p, x2p, x1p, x1p], [y1p, y1p, y2p, y2p, y1p], color='b', linewidth=3.0, label='predicted')

left_intersect = max(x1, x1p) if x1p < x2 else 0
right_intersect = min(x2, x2p) if x1p < x2 else 0
bottom_intersect = max(y1, y1p) if y1p < y2 else 0
top_intersect = min(y2, y2p) if y1 < y2p else 0

if (right_intersect - left_intersect)*(top_intersect - bottom_intersect) > 0:
    plt.fill_between(np.linspace(left_intersect, right_intersect, 10), 
                     bottom_intersect, 
                     top_intersect, 
                     hatch='///', 
                     color = 'w', 
                     edgecolor="r", 
                     linewidth=0.0)

fnfp = {'FN': {'x': None, 
               'y': None}, 
        'FP': {'x': None, 
               'y': None}}

if x1 < left_intersect and left_intersect < x2 and x1p < right_intersect and right_intersect < x2p:
    fnfp['FN']['x'] = np.mean(sorted([x1, x2, x1p, x2p])[:2])
    fnfp['FP']['x'] = np.mean(sorted([x1, x2, x1p, x2p])[-2:])
elif x1p < left_intersect and left_intersect < x2p and x1 < right_intersect and right_intersect < x2:
    fnfp['FN']['x'] = np.mean(sorted([x1, x2, x1p, x2p])[-2:])
    fnfp['FP']['x'] = np.mean(sorted([x1, x2, x1p, x2p])[:2])
else:
    fnfp['FN']['x'] = (x1 + x2) / 2
    fnfp['FP']['x'] = (x1p + x2p) / 2
    
if y1 < bottom_intersect and bottom_intersect < y2 and y1p < top_intersect and top_intersect < y2p:
    fnfp['FN']['y'] = np.mean(sorted([y1, y2, y1p, y2p])[:2])
    fnfp['FP']['y'] = np.mean(sorted([y1, y2, y1p, y2p])[-2:])
elif y1p < bottom_intersect and bottom_intersect < y2p and y1 < top_intersect and top_intersect < y2:
    fnfp['FN']['y'] = np.mean(sorted([y1, y2, y1p, y2p])[-2:])
    fnfp['FP']['y'] = np.mean(sorted([y1, y2, y1p, y2p])[:2])
else:
    fnfp['FN']['y'] = (y1 + min(y2, top_intersect, bottom_intersect)) / 2 #(y1 + y2) / 2
    fnfp['FP']['y'] = (y1p + min(y2p, top_intersect, bottom_intersect)) / 2 #(y1p + y2p) / 2
    
if (right_intersect - left_intersect)*(top_intersect - bottom_intersect) > 0:
    plt.text((left_intersect+right_intersect)/2, (top_intersect+bottom_intersect)/2, 'TP', fontsize='x-large')

plt.text(fnfp['FP']['x'], fnfp['FP']['y'], 'FP', fontsize='x-large')
plt.text(fnfp['FN']['x'], fnfp['FN']['y'], 'FN', fontsize='x-large')
    
plt.text(min(x1, x1p, x2, x2p)-1, max(y1, y1p, y2, y2p)+1, 'TN', fontsize='x-large')

plt.xlim(left = min(x1, x1p, x2, x2p)-3)
plt.ylim(top = max(y1, y1p, y2, y2p)+3)
plt.legend();
```

</details>

{::options parse_block_html="false" /}
    
![png](/assets/images/Jaccard_object_detection_files/Jaccard_object_detection_21_0.png)
    


Such overlap between a pair of boxes are utilized in three different scenarios. When computing the loss the complement of this overlap is considered as the loss between the actual location and the predicted location of the object. When applied in the NMS method, this overlap indicates whether two predicted boxes are more or less same and one of them should be filtered out of the final prediction. When calculating the mAP, this represents the similarity between the actual bounding box and the predicted bounding box.

Jaccard Index can also be expressed in terms of the values in the contingency table (confusion matrix), where TP are the overlapping area, FP and FN are the remaining elements in the two boxes, and TN are the elements that are not in these three areas.

$$
\begin{align*}
JI &= \frac{TP}{TP+FP+FN} \\
Here, \\
TP &= \text{ True Positives } \\
FP &= \text{ False Positives }\\
FN &= \text{ False Negatives } \\
TN &= \text{ True Negatives }\\
\end{align*}
$$

This expression helps to measure the similarty between two sets of elements in the classification task. Jaccard Distance, which measures the dissimilarity between two sets, is expressed as (1-IoU).

## References & Acknowledgements

1. [The distribution of the flora in the Alpine zone.](https://nph.onlinelibrary.wiley.com/doi/10.1111/j.1469-8137.1912.tb05611.x)
2. [Edge detection and ridge detection with automatic scale selection](http://kth.diva-portal.org/smash/record.jsf?pid=diva2%3A452310&dswid=-3446)
3. [Create YOLOv3 using PyTorch from scratch](https://numbersmithy.com/create-yolov3-using-pytorch-from-scratch-part-1/)
4. [Intersection over Union (IoU): Definition, Calculation, Code](https://www.v7labs.com/blog/intersection-over-union-guide#:~:text=The%20intersection%20area%20of%20the,to%20determine%20the%20IoU%20score.)
5. [Intersection over Union (IoU) for object detection](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)
6. [Cloudfactory Intersection over Union (IoU)](https://wiki.cloudfactory.com/docs/mp-wiki/metrics/iou-intersection-over-union)
7. [IOU (Intersection over Union)](https://medium.com/analytics-vidhya/iou-intersection-over-union-705a39e7acef)
8. [Intersection over union (IoU) calculation for evaluating an image segmentation model](https://towardsdatascience.com/intersection-over-union-iou-calculation-for-evaluating-an-image-segmentation-model-8b22e2e84686)
9. [Intersection over Union (IoU) in Object Detection & Segmentation](https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/)
10. [Intersection Over Union for Object Detection](https://www.baeldung.com/cs/object-detection-intersection-vs-union)
11. [Intersection over Union (IoU): A comprehensive guide](https://machinelearningspace.com/intersection-over-union-iou-a-comprehensive-guide/)
12. [Intersection over Union for object detection](https://www.superannotate.com/blog/intersection-over-union-for-object-detection)
13. [Arrows between subplots](https://stackoverflow.com/questions/60807792/arrows-between-subplots)
14. [Matplotlib Annotations](https://matplotlib.org/stable/users/explain/text/annotations.html)
15. [A Comprehensive Guide to Inset Axes in Matplotlib](https://towardsdatascience.com/a-comprehensive-guide-to-inset-axes-in-matplotlib-87400e00a4e5)
