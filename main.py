import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()  #ascunde fereastra tk


image_PATH=askopenfilename(title="Selecteaza prima imagine",filetypes=[("imagini","*.exr"),("imaginipng","*.png")])
image_PATH2=askopenfilename(title="Selecteaza prima imagine",filetypes=[("imagini","*.exr"),("imaginipng","*.png")])

if(image_PATH):
    image=cv2.imread(image_PATH,cv2.IMREAD_UNCHANGED)
    image2=cv2.imread(image_PATH2,cv2.IMREAD_UNCHANGED)
    difference = cv2.absdiff(image , image2)
    EROARE_MEDIE=np.mean(difference)
    img=cv2.normalize(difference,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)
    #plt.imshow(img)
    #cv2.imshow("imagine",image)
    #cv2.imshow("img",img)
    #cv2.imshow("difference",img)
    plt.hist(difference.ravel(), bins=100, range=(0, 5000), color='blue', alpha=0.7)
    
plt.title("histograma harta de adancime")
plt.xlabel("adancime")
plt.ylabel("numarul de pixeli")
plt.grid(True)
plt.show()
cv2.waitKey(0)