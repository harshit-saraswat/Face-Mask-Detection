# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 13:44:48 2020

@author: ACER
"""

import cv2
import numpy as np
import tensorflow.keras
from PIL import Image, ImageOps

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

prototxtPath = "./models/deploy.prototxt"
weightsPath = "./models/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNetPath="./models/keras_model.h5"
maskNet=tensorflow.keras.models.load_model(maskNetPath)


def detectMask(frame,maskNet):
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    
    im_pil = Image.fromarray(frame)
    image = ImageOps.fit(im_pil, size, Image.ANTIALIAS)
    #turn the image into a numpy array
    image_array = np.asarray(image)
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    
    # Load the image into the array
    data[0] = normalized_image_array
    cv2.imshow("norm image",normalized_image_array)    
    # run the inference
    prediction = maskNet.predict(data)
    pred="No Mask"
    if prediction[0][0]>0.9:#prediction[0][1]:
        pred="With Mask"
        print("With Mask")
    else:
        pred="No Mask"
        print("No Mask")
    print(prediction)
    return pred,normalized_image_array

cam=cv2.VideoCapture(0)
fc=0
while True:
    _,frame=cam.read()
    frame=cv2.flip(frame,1)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
         (104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
    fc+=1
    ##loop over the detections
    prevArea=0
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
        
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.7:
              # compute the (x, y)-coordinates of the bounding box for
              # the object
              box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
              (startX, startY, endX, endY) = box.astype("int")
              
              
              if startX<=0:
                  startX=0
              elif startX>=w:
                  startX=w
              else:
                  pass
              if startY<=0:
                  startY=0
              elif startY>=h:
                  startY=h
              else:
                  pass
              if endX<=0:
                  endX=0
              elif endX>=w:
                  endX=w
              else:
                  pass
              if endY<=0:
                  endY=0
              elif endY>=h:
                  endY=h
              else:
                  pass
              
              try:  
                  face=frame[startY-20:endY+20,startX-50:endX+50]
                  area=(endX-startX)*(endY-startY)
                  cv2.imshow("Face",face)
                  pred,normFace=detectMask(face,maskNet)
                  cv2.rectangle(frame,(startX,startY),(endX,endY),(255,0,0),2,1)
                  
                  if pred=="No Mask":              
                      cv2.putText(frame,pred,(startX-10,startY-10),2,0.7,(0,0,255),2,1)
                  else:
                      cv2.putText(frame,pred,(startX-10,startY-10),2,0.7,(0,255,0),2,1)
              except:
                  continue
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==27:
        break
    if k==ord('k'):
        cv2.imwrite('./frame'+str(fc)+'.jpeg',frame)
        cv2.imwrite('./face'+str(fc)+'.jpeg',face)
        cv2.imwrite('./normFace'+str(fc)+'.jpeg',normFace)
    
cam.release()
cv2.destroyAllWindows()