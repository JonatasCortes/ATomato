from atomato import TomatoWindow, TomatoElement
import time
import cv2
import mouse

url = "http://uitestingplayground.com/progressbar"
window = TomatoWindow(url)

window.open()
window.wait_stability()
screen = window.screen_shot()

vertical_slice = slice(520, screen.get_height()-450)
horisontal_slice = slice(50, screen.get_width()-20)

sliced = screen[vertical_slice, horisontal_slice]
start, stop, _, _, progess_bar = sliced.list_elements()

start.click()
time.sleep(3)
stop.click()






# Refazer Utils.find_element_in_screen() e element.wait_visibility()



# Start: 108, Stop: 109, progess_bar: 112