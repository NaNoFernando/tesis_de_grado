from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
import os
import numpy as np
from distribucion import entrenamiento
from tensorflow import keras
from redNeuronal import *
import tkinter.font as font

def video_de_entrada():
    global cap
    global count
    if selected.get() == 1:
        btnEnd.configure(state="active")
        btnCaptura.configure(state="active")
        rad1.configure(state="disabled")
        rad2.configure(state="disabled")
        rad3.configure(state="disabled")
        lblInfoVideoPath.configure(text="Capturando Video ...")
        count =0
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        visualizar()
    if selected.get() == 2:
        btnEnd.configure(state="active")
        btnCaptura.configure(state="disabled")
        rad1.configure(state="disabled")
        rad2.configure(state="disabled")
        rad3.configure(state="disabled")
        lblInfoVideoPath.configure(text="Entrenando Modelo ...")
        entrenando()
    if selected.get() == 3:
        btnEnd.configure(state="active")
        btnCaptura.configure(state="disabled")
        rad1.configure(state="disabled")
        rad2.configure(state="disabled")
        rad3.configure(state="disabled")
        lblInfoVideoPath.configure(text="Reconocimiento ...")
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        probar()
        
def finalizar_limpiar():
    lblVideo.image = " "
    lblVideo2.image = " "
    lblInfoVideoPath.configure(text="")
    btnCaptura.configure(state="disabled")
    btnEnd.configure(state="disabled")
    rad1.configure(state="active", command=video_de_entrada,bg=fondo)
    rad2.configure(state="active")
    rad3.configure(state="active")
    texto.delete(0,"end")
    texto2.delete(0,"end")
    selected.set(0)
    cap.release()
    
def visualizar():
    global cap
    global ret
    global frame
    global image
    x=20
    y=15
    w=240
    h=180
    ret, frame = cap.read()
    if ret == True:
        frame = imutils.resize(frame, width=280)
        
        #frame = deteccion_facilal(frame)
        image=frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        frame = cv2.rectangle(frame, (x+50, y+50), (x + w-50, y + h-50), (0, 0, 255), 2)
        frame = cv2.circle(frame, (140, 110), 5, (255, 0, 0), 3)
        #frame esta en array lo pasamos a imagen
        im = Image.fromarray(frame)
        #im esta en estado de MEMORIA tipo imagen pero lo pasamos a Imagen normal
        img = ImageTk.PhotoImage(image=im)
        lblVideo.configure(image=img)
        lblVideo.image = img
        lblVideo.after(5, visualizar)
    else:
        lblVideo.image = ""
        lblInfoVideoPath.configure(text="")
        rad1.configure(state="active")
        rad2.configure(state="active")
        rad3.configure(state="active")
        selected.set(0)
        btnEnd.configure(state="disabled")
        cap.release()
        

def capturar():
    global image
    global count
    personName = texto.get()
    personCi=texto2.get()
    if personName!="" and personCi!="":
        dataPath = 'usuarios' 
        personPath = dataPath + '/' + personCi+"_"+personName
        if not os.path.exists(personPath):
            print('Carpeta creada: ',personPath)
            os.makedirs(personPath)
        imageMuestra=image.copy()
        image = cv2.resize(image,(224,224),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/img_{}.jpg'.format(count),image)
        count = count + 1
    else:
        image=""
        lblInfoVideoPath.configure(text="Inserte un Nombre y C.I de usuario para continuar ...")
    im = Image.fromarray(imageMuestra)
    img = ImageTk.PhotoImage(image=im)
    lblVideo2.configure(image=img)
    lblVideo2.image = img
    
def probar():
    global modelo
    ruta=filedialog.askopenfilename(title="Seleccionar Imagen",filetypes=(("Imagenes JPG","*.jpg"),("TODO","*.*")))
    I=cv2.imread(ruta)
    modelo=keras.models.load_model("entrenamiento/mi_modelo")
    copia=I.copy()
    copia = cv2.cvtColor(copia, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(copia)
    img = ImageTk.PhotoImage(image=im)
    lblVideo.configure(image=img)
    lblVideo.image = img
    if round(modelo.predict(np.array([I]))[0][0])==1:
        print("La lesion es cancer")
        lblInfoVideoPath.configure(text="La lesion es melanoma")    
    else:
        print("La lesion es Benigna")
        lblInfoVideoPath.configure(text="La lesion es benigna") 


def cerrar():
    root.quit()
    print("Cerrando ventana :)")
    root.destroy()

def entrenando():
    global modelo
    lblInfoVideoPath.configure(text="Entrenando, Por Favor Espere...")   
    print("entrenando espere....") 
    modelo=entrenamiento()
    lblInfoVideoPath.configure(text="Modelo Entrenado Con Exito ...")
    

cap = None
count =0
ret,frame,image="","",""
root = Tk()
root.geometry("570x410")
root.resizable(width=False,height=False)
root.title("Sistema Detector de Melanomas")
fondo="light blue"
root.configure(bg=fondo)
font.nametofont("TkFixedFont").configure(size = 12, underline = True,family="Helvetica")

lblInfo1 = Label(root, text=" Detector de Melanomas ",bg=fondo, font = ("Times New Roman",15,"bold"))
lblInfo1.grid(column=0, row=0, columnspan=3)

selected = IntVar()
rad1 = Radiobutton(root, text="Capturar", width=20, value=1, variable=selected, command=video_de_entrada,bg=fondo)
rad2 = Radiobutton(root, text="Entrenar", width=10, value=2, variable=selected, command=video_de_entrada,bg=fondo)
rad3 = Radiobutton(root, text="Reconocer", width=20, value=3, variable=selected, command=video_de_entrada,bg=fondo)
rad1.grid(column=0, row=1, pady=10)
rad2.grid(column=1, row=1, pady=10)
rad3.grid(column=2, row=1, pady=10)

texto=Entry(root)
texto.grid(column=1, row=2, columnspan=2)

lblInfo2 = Label(root, text="Nombre de Usuario :",bg=fondo)
lblInfo2.grid(column=0, row=2, columnspan=2)

lblInfo3 = Label(root, text="C.I. del usuario :",bg=fondo)
lblInfo3.grid(column=0, row=3, columnspan=2)

texto2=Entry(root)
texto2.grid(column=1, row=3, columnspan=2)

lblInfoVideoPath = Label(root, text="  Introdusca el nombre del paciente y su C.I para poder crear su carpeta personal  ", width=60,bg=fondo,font = ("Times New Roman",11,"bold"),fg="green")
lblInfoVideoPath.grid(column=0, row=4, columnspan=3)

lblVideo = Label(root,bg=fondo)
lblVideo.grid(column=0, row=5, columnspan=2)

lblVideo2 = Label(root,bg=fondo)
lblVideo2.grid(column=2, row=5, columnspan=2)


btnEnd = Button(root, text="Finalizar Captura y Limpiar", state="disabled", command=finalizar_limpiar)
btnEnd.grid(column=0, row=6, columnspan=2, pady=10)

btnCaptura = Button(root, text="Capturar Fotografia",state="disabled", command=capturar)
btnCaptura.grid(column=2, row=6, columnspan=2)



root.mainloop()