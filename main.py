import cv2.cv2 as cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import tensorflow_hub as hub

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
root = tk.Tk()
label = tk.Label(root)
image = 0
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


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
    image.save(filename.name)


def show_img():
    img = ImageTk.PhotoImage(image=image)

    label.config(image=img)
    label.image = img
    label.grid(row=0, column=0)

    save_button = tk.Button(root, text='save', command=save_file)
    save_button.grid(row=3, column=0)


def face_overlay():
    filename = filedialog.askopenfilename()
    global image
    image = face_swap(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    show_img()


def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def change(file_img, file_style):
    img = load_img(file_img)
    style = load_img(file_style)
    img = hub_model(tf.constant(img), tf.constant(style))[0]
    return tensor_to_image(img)


def change_style():
    filename_img = filedialog.askopenfilename()
    filename_style = filedialog.askopenfilename()
    global image
    image = change(filename_img, filename_style)
    show_img()


def gui():
    button_face = tk.Button(root, text='face overlay', command=face_overlay)
    button_face.grid(row=1, column=0)
    button_style = tk.Button(root, text='change style', command=change_style)
    button_style.grid(row=2, column=0)
    root.mainloop()


if __name__ == '__main__':
    gui()
