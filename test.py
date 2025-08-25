import cv2
from atomato import TomatoElement
from atomato import TomatoUtils
from atomato import TomatoImage
from atomato import TomatoScreen
from atomato import TomatoWindow
import time
import numpy as np

test = TomatoElement("velocity_files\\playground.png")
m1 = test.get_image_matrix()

screen = TomatoScreen("velocity_files\\test_screen.png")
screen = screen[510:700]

bah = screen.find_element(test)
bah.show()



