import cv2 as cv

video=cv.VideoCapture(0)
licence_cascade=cv.CascadeClassifier("haarcascade_russian_plate_number.xml")
# CascadeClassifier method in cv2 module supports the loading of haar-cascade XML files. 

while True:
        _,img=video.read()
        
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        # canny = cv.Canny(img, 125, 175)
        licences=licence_cascade.detectMultiScale(gray,1.1,6)
       

        for (x,y,w,h)in licences:
            cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv.putText(img,"LICENCE PLATE",(x,y),cv.FONT_HERSHEY_COMPLEX,1,(24,24,23),2)
            imroi=img[y:y+h,x:x+w]
            canny = cv.Canny(imroi,125,175)
            cv.imshow("ROI",canny)
            
        cv.imshow('img',img)
        if cv.waitKey(1) == ord('q'):  
            cv.imwrite("q.jpg",canny)
            cv.rectangle(img,(0,200),(640,300),(76,155,78),cv.FILLED)
            cv.putText(img,"SCAN SAVED",(15,240),cv.FONT_HERSHEY_COMPLEX,1,(0,255,255),1)
            cv.imshow("img",img)
            cv.waitKey(500)
            break
    
video.release()
cv.destroyAllWindows()

