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


import cv2
import numpy as np
from os import listdir
from os.path import isfile,join
import os
import time
import threading
from sklearn.externals import joblib
import pickle
from pprint import pprint


import pandas
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier




'''data_path='faces/16bcs042'
onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))]

training_data,labels=[],[]

for i,files in enumerate(onlyfiles):
    image_path=data_path+ '/' +onlyfiles[i]
    print(image_path)
    
    images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    training_data.append(np.asarray(images,dtype=np.uint8))
    labels.append(i)


labels=np.asarray(labels,dtype=np.int32)



model = cv2.createLBPHFaceRecognizer()

model.train(np.asarray(training_data),np.asarray(labels))
print("model trained sucessfully")

'''
model=None


face_classifier=cv2.CascadeClassifier(r"H:\python project\udemy_opencv\udemy_opencv_tutorials\Haarcascades\haarcascade_frontalface_default.xml")

subjects = ["", "16bcs042","16bcs011"]


def face_detector(image):
    roi=None
    #gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(image,1.3,5)
    
    print("detected")
    
    
    if faces is None:
        return None
        print(" no face")
        
   
    
    for (x,y,w,h) in faces:
        
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
        
        roi=image[y:y+h,x:x+h]
        roi=cv2.resize(roi,(200,200))
        
        
        
        
   
    return image,roi,faces

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                #Data selection
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def training_data(folder_path):
    
    
    dirs = os.listdir(folder_path)
    
    
    faces=[]
   
    labels=[]
    count1=0
    count2=0

    print("2")
    print(folder_path)
    
    for folder_name in dirs:
        print("below folder")
        print(folder_name)
        print(dirs)
        
       
        
        print("3")
        
       
        #label = int(folder_name.replace(folder_name[0:1],""))

        label=folder_name
        
        #label=str(folder_name)
  
        
       
        image_folder_path = folder_path + "/" + folder_name
        
        print(image_folder_path)
        

        
        folder_images_list = os.listdir(image_folder_path)
        print(folder_images_list)
        
      
        for image_name in folder_images_list:
            print(image_name)
            
           
            '''if image_name.startswith("."):
                continue;'''
            
           
            image_path = image_folder_path + "/" + image_name
            print(image_path)
            
           
            image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

            print(image)
            

            
           
            cv2.imshow("training on image...", image)
            

            
            cv2.waitKey(100)
           
            print(image)
            
            
           

            print(image.shape)
            one_face_array=np.asarray(image,dtype=np.uint8)
            print(one_face_array.shape)
            
            
            x=np.reshape(one_face_array,(np.product(one_face_array.shape),))
            print(x.shape)
           
            
            faces.append(x)
            
            
            labels.append(label)
            

    print("face count is ",len(faces))
    print("label count is ",len(labels))
    print(labels)

    pprint(faces)
    
    
    
   
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                        #MODEL TRAINING
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def model_training(train_img,labels):

    '''data_path='faces/16bcs042'
    onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))]

    training_data,labels=[],[]

    for i,files in enumerate(onlyfiles):
        image_path=data_path+ '/' +onlyfiles[i]
        images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        training_data.append(np.asarray(images,dtype=np.uint8))
        labels.append(i)


    labels=np.asarray(labels,dtype=np.int32)

''' 
    
    model=RandomForestClassifier(random_state=42)
    
    
    #model = cv2.createLBPHFaceRecognizer()
    print("before")
    
    #training_image=np.asarray(train_img)

    training_image=train_img
    print(training_image)
    
    print("after")
    training_data=np.array(training_image)
    
    print(training_data[0])
    
    

    label=np.asarray(labels,dtype=np.uint32)
    print(label)
    
    
    
    '''print("dim1")
    print(np.shape(training_image))
    nsamples, nx, ny = np.shape(training_image)
    training_data = training_image.reshape((nsamples,nx*ny))
    print("dim2")
    print(np.shape(training_data))
    print(training_data)'''
    #model.train(training_image,np.asarray(labels))
    
    model.fit(training_data,label)
    
    print(model)
    print("model trained sucessfully")
   
    
    joblib_file = "saved_model/joblib_model.pkl"  
    joblib.dump(model, joblib_file)
    
    
    
    

    print("model saved successfully")
    
    
    ''' model.train(train_img,np.array(labels))
    print("here")
    result=PIL.Image.fromarray(train_img[0])
    result.show()
    time.sleep(100)'''

    
    



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                        # MAIN PROGRAM
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def main():
    print("Preparing data...")
    #"faces"--> folder path
    faces, labels = training_data("hand_image")
    print("Data prepared")
    #labels are unique id like 42 of 16bcs042 for the faces
    print(faces,labels)

    print("Training model")
    
    model_training(faces,labels)
    print(" Model trained")

  
if __name__=="__main__":
    main()