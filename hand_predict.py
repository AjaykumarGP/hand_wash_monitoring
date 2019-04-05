import cv2
import numpy as np 
import time
i=0

import pandas
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib








##############################
cap=cv2.VideoCapture(0)
count=0

while(True):

    
    


    ret,frame=cap.read()
    hand=cv2.resize(frame,(200,200))
    gray=cv2.cvtColor(hand,cv2.COLOR_BGR2GRAY)
    
    
    
    
    '''lower_blue=np.array([0,10,60])
    upper_blue=np.array([20,150,255])

    mask=cv2.inRange(gray,lower_blue,upper_blue)
    res=cv2.bitwise_and(frame,frame,mask=mask)

     

    refill_blacks(mask)

    resized_image=cv2.resize(gray,(133,100))
    resized_image=resized_image.flaten("F").reshape(1,133*100)'''

   
    load_=joblib.load("saved_model/joblib_model.pkl")
    
    

    x=np.reshape(gray,(np.product(gray.shape),))

   

    x=x.reshape(1,-1)

    
    '''nsamples, nx,ny= np.shape(frame)
    training_data=frame.reshape((nsamples,nx*ny))
    print(training_data.shape)
    print(nsamples)
    print(nx)
    print(ny)'''
    
    
    prediction= load_.predict(x)
    print("below prediction")
    print(prediction)
    
    
    

    if(prediction==0):
        gesture=""
    elif(prediction==1):
        gesture="first step"
    elif(prediction==2):
        gesture="second step"
    elif(prediction==3):
        gesture="Third step"
    elif(prediction==4):
        gesture="Forth step"
    elif(prediction==5):
        gesture="Fifth step"
    elif(prediction==6):
        gesture="Sixth step"
    
    cv2.putText(frame,"Gesture="+str(gesture),(1,250),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)
    cv2.imshow("res",frame)

    count=count+1

    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()