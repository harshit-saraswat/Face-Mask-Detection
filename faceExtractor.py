import cv2
import numpy as np
import os
prototxtPath = "deploy.prototxt"
weightsPath = "res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")

images=[]
for image in os.listdir("./with_mask"):
    print(image)
    images.append("./with_mask/"+image)
    imgPath="./with_mask/"+image
    img=cv2.imread(imgPath)
    cv2.imshow("image",img)
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300),
         (104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
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
              # # ensure the bounding boxes fall within the dimensions of
              # # the frame
              # (startX, startY) = (max(0, startX), max(0, startY))
              # (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
              print(confidence)
              print(startX,startY,endX,endY)
              face=img[startY:endY,startX:endX]
              area=(endX-startX)*(endY-startY)
              if prevArea<area:
                  prevArea=area
                  try:
                      cv2.imshow("Face",face)
                      cv2.imwrite("./withMask/"+image,face)
                      
                  except:
                      continue
                  
    cv2.waitKey(1)    
    cv2.destroyAllWindows()
# vs = cv2.VideoCapture(0)
# # loop over the frames from the video stream
# while True:
#     # grab the frame from the threaded video stream and resize it
#     # to have a maximum width of 400 pixels
#     ret,frame = vs.read()
# for image in os.listdir('./without_mask/'):
#     imgPath='./without_mask/'+image
#     frame=cv2.imread(imgPath)
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
#     (104.0, 177.0, 123.0))

#     # pass the blob through the network and obtain the face detections
#     faceNet.setInput(blob)
#     detections = faceNet.forward()

#     # initialize our list of faces, their corresponding locations,
#     # and the list of predictions from our face mask network
#     faces = []
#     locs = []
#     preds = []

#     # loop over the detections
#     for i in range(0, detections.shape[2]):
#         # extract the confidence (i.e., probability) associated with
#         # the detection
#         confidence = detections[0, 0, i, 2]

#         # filter out weak detections by ensuring the confidence is
#         # greater than the minimum confidence
#         if confidence > 0.5:
#             # compute the (x, y)-coordinates of the bounding box for
#             # the object
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")

#             # ensure the bounding boxes fall within the dimensions of
#             # the frame
#             (startX, startY) = (max(0, startX), max(0, startY))
#             (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

#             # extract the face ROI, convert it from BGR to RGB channel
#             # ordering, resize it to 224x224, and preprocess it
#             face = frame[startY:endY, startX:endX]
#             cv2.imshow("Face",face)
#             cv2.imwrite("./withoutMask/"+image,face)

#     
#     # show the output frame
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(100) & 0xFF

#     # if the `q` key was pressed, break from the loop
#     if key == ord("q"):
#         break

# # do a bit of cleanup
# # vs.release()
# cv2.destroyAllWindows()