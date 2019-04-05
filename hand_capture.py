
import cv2
import numpy as np
import os
import time


'''import cv2
import numpy as np

cap=cv2.VideoCapture(0)

hand_cascade=cv2.CascadeClassifier("hand.xml")

while(True):
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    hands=hand_cascade.detectMultiScale(gray,1.1,5)

    for (x,y,w,h) in hands:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("frame",frame)

    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()'''


def create_folder(dir):
    try:
        if not os.path.exists(dir):
            os.mkdir(dir)
            
    except OSError:
        print("error in creating directory "+dir )




roll=raw_input("Enter roll number")
time.sleep(1)
cap=cv2.VideoCapture(0)

count=0
create_folder('hand_image/{}'.format(roll))

x1=300
y1=300
x2=100
y2=100
time.sleep(10)
while True:
    time.sleep(0.2)
    count+=1

    ret,frame=cap.read()
    print("Image collected {}".format(count))
    #cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),0)

    #cropped_hand=frame[y1:y1+y2,x1:x1+x2]

    hand=cv2.resize(frame,(200,200))
    hand=cv2.cvtColor(hand,cv2.COLOR_BGR2GRAY)

    file_name_path=('./hand_image/' + roll + '/{}-'.format(roll) + str(count) + '.jpg')
    cv2.imwrite(file_name_path,hand)
    cv2.putText(hand,str(count),(50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow("face cropper",hand)


    if cv2.waitKey(1)==13 or count==100:
        break

cap.release()
cv2.destroyAllWindows()
print("collected complete")