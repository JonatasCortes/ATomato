from atomato import TomatoElement
from atomato import TomatoImage
from atomato import TomatoUtils
import time

#TEST URL: http://uitestingplayground.com/disabledinput

change = TomatoElement("visibility_files\\playground.png")
enable = TomatoElement("visibility_files\\enable.png")
input_enabled = TomatoElement("visibility_files\\input_enabled.png")

change.update_coordinates(under_step=22, cutoff=14)
change.move_center(y_delta=change.get_height())
change.delete_input()
change.input("Davi Brito")

enable.update_coordinates(under_step=10, cutoff=14)
enable.click()

input_enabled.wait_visibility(under_step=12, cutoff=15)

change.delete_input()
change.input("baguga")




