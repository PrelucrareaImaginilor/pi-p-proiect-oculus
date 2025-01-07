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
            img = cv2.imread(cale_fisier, cv2.IMREAD_GRAYSCALE)  # Citeste imaginea in grayscale
            if img is not None:
                imagini.append(img)
    return imagini

def aliniaza_imagine(imagine, referinta):
    if imagine.shape[:2] != referinta.shape[:2]:
        imagine = cv2.resize(imagine, (referinta.shape[1], referinta.shape[0]))
    return imagine

def oglindeste_imagine(imagine):
    return cv2.flip(imagine, 1)

def proceseaza_si_oglindeste_imagini(folder, referinta):
    imagini = incarca_imagini_din_folder(folder)
    imagini_corectate = [aliniaza_imagine(oglindeste_imagine(img), referinta) for img in imagini]
    return imagini_corectate

def calculeaza_media_imaginilor(imagini):
    if not imagini:
        raise ValueError("Nu exista imagini pentru a calcula media.")
    imagine_medie = np.mean(np.stack(imagini, axis=0), axis=0)
    return imagine_medie.astype(imagini[0].dtype)

def detecteaza_schimbari(imagine1, imagine2, prag):
    diferenta = cv2.absdiff(imagine1, imagine2)
    schimbari = (diferenta > prag).astype(np.uint8) * 255
    return schimbari

def proceseaza_foldere_si_calculeaza_diferente(folder1, folder2):
    imagini2 = incarca_imagini_din_folder(folder2)
    if not imagini2:
        raise ValueError("Folderul al doilea este gol sau nu contine imagini valide.")

    referinta = imagini2[0]
    imagini1 = proceseaza_si_oglindeste_imagini(folder1, referinta)

    imagine_medie1 = calculeaza_media_imaginilor(imagini1)
    imagine_medie2 = calculeaza_media_imaginilor(imagini2)

    if imagine_medie1.shape != imagine_medie2.shape:
        imagine_medie2 = cv2.resize(imagine_medie2, (imagine_medie1.shape[1], imagine_medie1.shape[0]))

    scor_ssim, diferenta = ssim(imagine_medie1, imagine_medie2, full=True)
    print(f"Scor SSIM (imagini medii): {scor_ssim:.4f}")

    diferenta = (diferenta * 255).astype("uint8")

    schimbari = detecteaza_schimbari(imagine_medie1, imagine_medie2, prag=10)

    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    plt.title("Imagine Medie 1")
    plt.imshow(imagine_medie1, cmap='gray')

    plt.subplot(2, 3, 2)
    plt.title("Imagine Medie 2")
    plt.imshow(imagine_medie2, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title("Diferenta (SSIM)")
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
folder1 = askdirectory(title="Selectati primul folder")
folder2 = askdirectory(title="Selectati al doilea folder")

proceseaza_foldere_si_calculeaza_diferente(folder1, folder2)
