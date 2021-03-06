import cv2
import os
import numpy as np
import faceRecogntion as fr

#for (x,y,w,h) in faces_detected:
#    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
    
#resized_img=cv2.resize(test_img,(1000,700))
#cv2.imshow("title",resized_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows

faces,faceID,=fr.labels_for_training_data('./trainingImg/')
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.save('./trainingData.yml')
name={0:"Shahruk", 1:"Priyanka"}

cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()
    faces_detected,gray_img=fr.faceDetection(test_img)

    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+h,x:x+w]
        label,confidence=face_recognizer.predict(roi_gray)
        print("confidence: ",confidence)
        print("label: ",label)
        if(confidence<=41):
            fr.draw_rect(test_img,face)
        else:
            continue
        predicted_name=name[label]
        fr.put_text(test_img,predicted_name,x,y)
        
    resized_img=cv2.resize(test_img,(1000,700))
    cv2.imshow("title",resized_img)
    if cv2.waitKey(10)==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows