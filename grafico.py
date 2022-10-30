from tensorflow import keras
import cv2
import numpy as np
from redNeuronal import *
from tkinter import *
from tkinter import filedialog
from distribucion import entrenamiento
from PIL import Image, ImageTk

def entrenar():
    global modelo
    modelo=entrenamiento()
    return modelo
def probar():
    global modelo
    ruta=filedialog.askopenfilename(title="Seleccionar Imagen",filetypes=(("Imagenes JPG","*.jpg"),("TODO","*.*")))
    I=cv2.imread(ruta)
    modelo=keras.models.load_model("entrenamiento/mi_modelo")
    if round(modelo.predict(np.array([I]))[0][0])==1:
        print("La lesion es cancer")
        cv2.imshow("Cancer",I)
    else:
        print("La lesion es Benigna")
        cv2.imshow("Benigna",I)
def cerrar():
    root.quit()
    print("Cerrando ventana :)")
    root.destroy()
def callback():
    pass
root=Tk()
root.geometry('200x200+800+500')
root.configure(background='dark turquoise')
root.protocol("WM_DELETE_WINDOW",cerrar)
root.title("Vision Artificial")
modelo=None

btnTrain = Button(root, text="Entrenamiento", command=entrenar)
btnTrain.grid(column=0, row=1, pady=10)

btnTest = Button(root, text="Prueba", command=probar)
btnTest.grid(column=1, row=1, pady=10)

label=Label(root)
label.grid(column=2,row=2,pady=10)

root.after(1,callback)

root.mainloop()
