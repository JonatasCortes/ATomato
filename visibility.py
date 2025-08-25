from atomato import TomatoElement
from atomato import TomatoImage
from atomato import TomatoUtils
from atomato import TomatoWindow
import cv2
import time

url = "http://uitestingplayground.com/disabledinput"
utils = TomatoUtils()

change = TomatoElement("visibility_files\\change.png")
enable = TomatoElement("visibility_files\\enable.png")
input_enabled = TomatoElement("visibility_files\\input_enabled.png")

window = TomatoWindow(url)
window.open()

change.wait_visibility(window, cutoff=15)
change.delete_input()
change.type("baguga")

enable.update_coordinates(window, cutoff=25)
enable.click()

window.wait_change()

change.delete_input()
change.type("kalala")

