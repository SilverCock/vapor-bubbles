import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('G:\\experement\\filtered\\5_wT_4_frames\\5_wT_4_frame006221.png')
if image is None:
    print("Error opening file")
    exit()  

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(
    src=gray, 
    ksize=(5, 5),  
    sigmaX=0     
)
edges = cv2.Canny(
    image=blurred,
    threshold1=50,   # Нижний порог
    threshold2=150   # Верхний порог
)
contours, hierarchy = cv2.findContours(

image=edges,

mode=cv2.RETR_TREE,

method=cv2.CHAIN_APPROX_SIMPLE

)
result = image.copy()
cv2.drawContours(
    image=result,
    contours=contours,
    contourIdx=-1,          
    color=(0, 255, 0),      # Зеленый цвет
    thickness=2        
)

plt.subplot(121),plt.imshow(image,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(result,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])


plt.show()