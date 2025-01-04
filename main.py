import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askdirectory

def incarca_imagini_din_folder(cale_folder):
    imagini = []
    for fisier in sorted(os.listdir(cale_folder)):
        cale_fisier = os.path.join(cale_folder, fisier)
        if fisier.endswith(".png"):
            img = cv2.imread(cale_fisier, cv2.IMREAD_UNCHANGED)
            if img is not None:
                imagini.append(img)
    return imagini

def calculeaza_media_imaginilor(imagini):
    if not imagini:
        raise ValueError("Nu exista imagini pentru a calcula media.")
    imagine_medie = np.mean(np.stack(imagini, axis=0), axis=0)
    return imagine_medie.astype(imagini[0].dtype)

def proceseaza_foldere_si_calculeaza_diferente(folder1, folder2):
    imagini1 = incarca_imagini_din_folder(folder1)
    imagini2 = incarca_imagini_din_folder(folder2)

    if not imagini1 or not imagini2:
        raise ValueError("Unul sau ambele foldere sunt goale sau nu contin imagini valide.")

    imagine_medie1 = calculeaza_media_imaginilor(imagini1)
    imagine_medie2 = calculeaza_media_imaginilor(imagini2)

    if imagine_medie1.shape != imagine_medie2.shape:
        imagine_medie2 = cv2.resize(imagine_medie2, (imagine_medie1.shape[1], imagine_medie1.shape[0]))

    scor_ssim, diferenta = ssim(imagine_medie1, imagine_medie2, full=True)
    print(f"Scor SSIM (imagini medii): {scor_ssim:.4f}")

    diferenta = (diferenta * 255).astype("uint8")

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 3, 1)
    plt.title("Imagine Medie 1")
    plt.imshow(imagine_medie1, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Imagine Medie 2")
    plt.imshow(imagine_medie2, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Diferenta (SSIM)")
    plt.imshow(diferenta, cmap='hot')

    plt.tight_layout()
    plt.show()

Tk().withdraw()
folder1 = askdirectory(title="Selectati primul folder")
folder2 = askdirectory(title="Selectati al doilea folder")

proceseaza_foldere_si_calculeaza_diferente(folder1, folder2)
