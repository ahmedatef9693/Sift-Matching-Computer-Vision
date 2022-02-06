import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

ImageA = cv.imread('query.jpg',0)
Folder_Name = 'tiny_data'
sift = cv.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(ImageA, None)
#Looping throu other images in a file

Distances = []
MinimumDistance = 0
currentimageindex = 0
for file in os.listdir(Folder_Name):

    current_path = os.path.join(Folder_Name,file)
    ImageB = cv.imread(current_path,0)
    kp2, des2 = sift.detectAndCompute(ImageB, None)
    bf = cv.BFMatcher()
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    DistanceSums = 0

    for i in range(10):
        DistanceSums = DistanceSums + matches[i].distance
    Distances.append(DistanceSums)
    print('Sum Of first 10 Distances for image : '+str(currentimageindex)+' = '+str(Distances[currentimageindex]))
    # Draw first 10 matches.
    output = cv.drawMatches(ImageA, kp1,
                           ImageB, kp2,
                           matches[0:10],
                           flags=2, outImg=None)

    plt.imshow(output,'gray'), plt.show()
    currentimageindex = currentimageindex+1


MinIndex = np.argmin(Distances)
MinimumDistance = min(Distances)
index = 0
for file in os.listdir(Folder_Name):
    if index == MinIndex:
        current_path = os.path.join(Folder_Name, file)
        Image = cv.imread(current_path, 0)
        cv.putText(Image,'Least Sum Image of distance = '+str(MinimumDistance),(255,255),cv.FONT_HERSHEY_TRIPLEX,fontScale=0.5,color=(255,0,0),thickness=1)
        cv.imshow('Least Sum of distances = '+str(MinimumDistance),Image)
        cv.waitKey(0)

    index = index + 1


