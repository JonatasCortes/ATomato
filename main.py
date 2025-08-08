from atomato import TomatoImage
from atomato import Tomato as tmt
import cv2

tomato = TomatoImage("tomato.jpeg")
tomato_section = TomatoImage("tomato_section.jpg")

cv2.imshow("bah", tomato_section.getImageMatrix())
cv2.waitKey(0)

tmt.find_element_in_screen(tomato_section, tomato)
