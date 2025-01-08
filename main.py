import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askdirectory

def incarca_imagini(cale):
    imagini = []
    for fisier in sorted(os.listdir(cale)):
        cale_fisier = os.path.join(cale_folder, fisier)
        if fisier.endswith(".png"):
            img = cv2.imread(cale_fisier, cv2.IMREAD_UNCHANGED)  # Citeste imaginea fara modificari de tip de date
            if img is not None:
                imagini.append(img)
    return imagini

def proceseaza_si_oglindesteimag(folder, referinta):   #nu stiu dc orbek ia imaginea in orglinda
    imagini = incarca_imagini(folder)
    imagini_corectate = [aliniaza_imagine(oglindeste_imagine(img), referinta) for img in imagini]
    return imagini_corectate

def calculeaza_media_imaginilor(imagini):
    if not imagini:
        raise ValueError("Eroare incarcare fisier ")
    imagine_med = np.mean(np.stack(imagini, axis=0), axis=0) #calculeaza media aritmetica a imaginilor
    return imagine_med.astype(imagini[0].dtype)

def detecteaza_schimbari(imagine1, imagine2, prag): 
    diferenta = cv2.absdiff(imagine1, imagine2)     #afiseaza zonele mai pronuntate
    schimbari = (diferenta > prag).astype(np.uint8) * 255
    return schimbari

def proceseaza_foldere(folder1, folder2):
    imagini2 = incarca_imagini(folder2) #preia toate fisierele dintr un folder si caluleaza o medie a valorilor
    if not imagini2:
        raise ValueError("Folderul al doilea este gol sau nu contine imagini valide.")

    referinta = imagini2[0]
    imagini1 = proceseaza_si_oglindesteimag(folder1, referinta)

    imagine_medie1 = calculeaza_media_imaginilor(imagini1)
    imagine_medie2 = calculeaza_media_imaginilor(imagini2)

    if imagine_medie1.shape != imagine_medie2.shape:
        imagine_medie2 = cv2.resize(imagine_medie2, (imagine_medie1.shape[1], imagine_medie1.shape[0]), interpolation=cv2.INTER_LINEAR)

    scor_ssim, diferenta = ssim(imagine_medie1, imagine_medie2, full=True, data_range=imagine_medie2.max() - imagine_medie2.min()) #ssim
    print(f"Scor SSIM (imagini medii): {scor_ssim:.4f}") #structural similarity index measure-detecteaza similaritatea dintre 2imagini

    diferenta = (diferenta * 255).astype("uint8") #afiseaza cu negru zonele cu diferente foarte mari
 
    schimbari = detecteaza_schimbari(imagine_medie1, imagine_medie2, prag=10)

    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    plt.title("Imagine1")
    plt.imshow(imagine_medie1, cmap='gray')

    plt.subplot(2, 3, 2)
    plt.title("Imagine2")
    plt.imshow(imagine_medie2, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title("Diferenta SSIM")
    plt.imshow(diferenta, cmap='hot')

    plt.subplot(2, 3, 4)
    plt.title("Schimbari Detectate")
    plt.imshow(schimbari, cmap='gray')

    plt.subplot(2, 3, 5)
    plt.title("Histograma Diferentelor")
    plt.hist(diferenta.ravel(), bins=50, color='blue', alpha=0.7)
    plt.xlabel("Valoarea Diferentei")
    plt.ylabel("Numarul de Pixeli")

    plt.tight_layout()
    plt.show()

Tk().withdraw()
folder = askdirectory(title="Selectati orbek folder")
folder2 = askdirectory(title="Selectati zedm folder")

proceseaza_foldere(folder, folder2)
