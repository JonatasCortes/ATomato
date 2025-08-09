from atomato import TomatoImage
from atomato import Tomato as tmt
import cv2

screen = TomatoImage("screen.jpg")
element1 = TomatoImage("element1.jpg")
element2 = TomatoImage("element2.jpg")
element3 = TomatoImage("element3.jpg")
element4 = TomatoImage("element4.jpg")

print(tmt.find_element_in_screen(element1, screen, under_step=1, cutoff=10))
print(tmt.find_element_in_screen(element2, screen, under_step=1, cutoff=10))
print(tmt.find_element_in_screen(element3, screen, under_step=1, cutoff=10))
print(tmt.find_element_in_screen(element4, screen, under_step=1, cutoff=10))
    