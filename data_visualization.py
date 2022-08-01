import cv2
import numpy as np
  

path = 'C:/Users/dqu/Desktop/CV_Project'

data = np.loadtxt('data.csv')

 
for i in range(data.shape[0]):
    face_array = data[i, :].reshape((48, 48)) # reshape
    cv2.imwrite(path + '//' + '{}.jpg'.format(i), face_array)
