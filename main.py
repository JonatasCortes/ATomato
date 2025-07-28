from atomato import TomatoImage
import cv2

tomato = TomatoImage("tomato.jpeg")

cv2.imwrite("tomato.png", tomato.getImageMatrix())