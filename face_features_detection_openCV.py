import numpy as np
import cv2

face_haarcascade = cv2.CascadeClassifier('haarcascade_eye.xml')
eye_haarcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_haarcascade = cv2.CascadeClassifier('haarcascade_smile.xml')

video = cv2.VideoCapture(0)
font=cv2.FONT_HERSHEY_COMPLEX_SMALL
while True:
    ret, img = video.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_haarcascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_haarcascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        i=0
        smile, rejectLevels, levelWeights = smile_haarcascade.detectMultiScale3(roi_gray, outputRejectLevels=True)
        for(ex,ey,ew,eh) in smile:
            if(round(levelWeights[i][0],3)>=3.5):
                cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
                cv2.putText(roi_color,str(round(levelWeights[i][0],3)),(ex,ey), font,1,(255,255,255),2)
            i+=1
                
    cv2.imshow('img',img)
    k = cv2.waitKey(60) & 0xff
    if k==27:
        break

video.release()
cv2.destroyAllWindows()
