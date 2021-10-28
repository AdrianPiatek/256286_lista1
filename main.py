import cv2.cv2 as cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

root = tk.Tk()
label = tk.Label(root)
image = 0


def overlay_image(background, foreground, alpha):
    foreground = foreground.astype(float)
    background = background.astype(float)
    alpha = alpha.astype(float)/255
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    out_image = cv2.add(foreground, background)
    return out_image


def face_swap(file):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread(file)
    face = cv2.imread('face.png')
    mask = cv2.imread('mask.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        img[y:y+h, x:x+w] = overlay_image(img[y:y+h, x:x+w], cv2.resize(face, (w, h)), cv2.resize(mask, (w, h)))
    return img


def save_file():
    filename = filedialog.asksaveasfile(mode='w', defaultextension=".png")
    if not filename:
        return
    cv2.imwrite(filename.name, image)


def get_file_name():
    filename = filedialog.askopenfilename()
    global image
    image = face_swap(filename)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(image=Image.fromarray(img))

    label.config(image=img)
    label.image = img
    label.grid(row=0, column=0)

    save_button = tk.Button(root, text='save', command=save_file)
    save_button.grid(row=2, column=0)


def gui():
    button = tk.Button(root, text='wybierz plik', command=get_file_name)
    button.grid(row=1, column=0)
    root.mainloop()


if __name__ == '__main__':
    gui()
