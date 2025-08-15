from atomato import TomatoImage
from atomato import Tomato as tmt
import time

start = time.time()

screen = TomatoImage("screen.jpg")
element1 = TomatoImage("element1.jpg")
element2 = TomatoImage("element2.jpg")
element3 = TomatoImage("element3.jpg")
element4 = TomatoImage("element4.jpg")

test_tomato = TomatoImage("tomato.jpeg")
test_tomato_section = TomatoImage("tomato_section.jpg")

#print(tmt().find_element_in_screen(element1, screen))
#print(tmt().find_element_in_screen(element2, screen))
#print(tmt().find_element_in_screen(element3, screen))
#print(tmt().find_element_in_screen(element4, screen))

end = time.time()
print(f"Tempo de execução: {end - start:.4f} segundos")

# region GABARITO
#print(tmt().find_element_in_screen(element1, screen, under_step=20, cutoff=8.62))
#print(tmt().find_element_in_screen(element2, screen, under_step=20, cutoff=0.9))
#print(tmt().find_element_in_screen(element3, screen, under_step=20, cutoff=2.4, match_resolutions=False))
#print(tmt().find_element_in_screen(element4, screen, under_step=21, cutoff=0.83))
# endregion