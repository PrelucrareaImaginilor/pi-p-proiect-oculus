import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from skimage.metrics import structural_similarity as ssim

Tk().withdraw()  #ascunde fereastra tk


image_PATH=askopenfilename(title="Selecteaza prima imagine",filetypes=[("imaginipng","*.png")])
image_PATH2=askopenfilename(title="Selecteaza prima imagine",filetypes=[("imaginipng","*.png")])

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
    if image.shape != image2.shape:
        print("Redimensionare imagini pentru a avea aceeași dimensiune...")
        image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))
    # 3. Calculul SSIM
    score, diff = ssim(image, image2, full=True)
    print(f"SSIM Score: {score:.4f}")

    # 4. Procesare diferențe
    diff = (diff * 255).astype("uint8")

    # 5. Vizualizare imagini și diferențe
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 3, 1)
    plt.title("Imagine 1")
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Imagine 2")
    plt.imshow(image2, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Diferență (SSIM)")
    plt.imshow(diff, cmap='hot')

    plt.tight_layout()
    plt.show()


plt.title("histograma harta de adancime")
plt.xlabel("adancime")
plt.ylabel("numarul de pixeli")
plt.grid(True)
plt.show()
cv2.waitKey(0)