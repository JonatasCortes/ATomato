from atomato import TomatoElement
from atomato import TomatoImage
from atomato import TomatoUtils
import time

#TEST URL: http://uitestingplayground.com/disabledinput

change = TomatoElement("visibility_files\\playground.jpg")
enable = TomatoElement("visibility_files\\enable.jpg")
disabled = TomatoElement("visibility_files\\disabled.jpg")

change.update_coordinates(under_step=12, cutoff=18.2)
change.move_center(y_delta=(-change.get_height()))
change.click()

#enable.update_coordinates(under_step=7, cutoff=24)
#enable.click()

disabled.wait_visibility(timeout=15)
#TESTAR MAIS WAIT_VISIBILITY

print("bah")




