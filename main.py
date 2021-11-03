import cv2.cv2 as cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf


root = tk.Tk()
label = tk.Label(root)
image = 0
hub_model = tensorflow_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


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
    face = cv2.imread('../256286_lista_1/face.png')
    mask = cv2.imread('../256286_lista_1/mask.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        img[y:y+h, x:x+w] = overlay_image(img[y:y+h, x:x+w], cv2.resize(face, (w, h)), cv2.resize(mask, (w, h)))
    return img


def save_file():
    filename = filedialog.asksaveasfile(mode='w', defaultextension=".png")
    if not filename:
        return
    image.save(filename.name)


def face_overlay():
    filename = filedialog.askopenfilename()
    global image
    image = face_swap(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    img = ImageTk.PhotoImage(image=image)

    label.config(image=img)
    label.image = img
    label.grid(row=0, column=0)

    save_button = tk.Button(root, text='save', command=save_file)
    save_button.grid(row=3, column=0)


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def change(file_img, file_style):
    img = tf.io.read_file(file_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    style = tf.io.read_file(file_style)
    style = tf.image.decode_image(style, channels=3)
    style = tf.image.convert_image_dtype(style, tf.float32)

    stylized_image = hub_model(tf.constant(img), tf.constant(style))[0]
    return tensor_to_image(stylized_image)


def change_style():
    filename_img = filedialog.askopenfilename()
    filename_style = filedialog.askopenfilename()
    global image
    image = change(filename_img, filename_style)


def gui():
    button_face = tk.Button(root, text='face overlay', command=face_overlay)
    button_face.grid(row=1, column=0)
    button_style = tk.Button(root, text='change style', command=change_style)
    button_style.grid(row=2, column=0)
    root.mainloop()


if __name__ == '__main__':
    gui()
