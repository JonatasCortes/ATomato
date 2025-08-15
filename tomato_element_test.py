from atomato import TomatoElement
from atomato import TomatoImage
from atomato import TomatoUtils
import time

#TEST URL: http://uitestingplayground.com/textinput?

test_element1 = TomatoElement("my_button.jpg")
test_element1.update_coordinates(under_step=14, cutoff=1.02)
test_element1.type("bah")
time.sleep(2)
test_element1.delete_input()
test_element1.type("tche")

test_element2 = TomatoElement("change_value_button.jpg")
test_element2.update_coordinates(under_step=20, cutoff=12.22)
test_element2.click()