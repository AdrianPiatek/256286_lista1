import cv2 as cv2
import numpy as np


def overlay_image(background, foreground, alpha):
    foreground = foreground.astype(float)
    background = background.astype(float)
    alpha = alpha.astype(float)/255
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    out_image = cv2.add(foreground, background)
    return out_image


def face_swap():
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Read the input image
    img = cv2.imread('man_face.jpeg')
    face = cv2.imread('face.png')
    mask = cv2.imread('mask.png')
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        img[y:y+h, x:x+w] = overlay_image(img[y:y+h, x:x+w], cv2.resize(face, (w, h)), cv2.resize(mask, (w, h)))
    # Display the output
    cv2.imshow('img', img)
    cv2.waitKey()


if __name__ == '__main__':
    face_swap()
