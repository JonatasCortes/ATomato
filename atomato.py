import numpy as np
import keyboard
import mouse
import cv2
from mss import mss
import os
import time

class TomatoImage:
    def __init__(self, image : str | np.ndarray):
        self.__setImage(image)

    def __black_white_image(self, image : np.ndarray) -> np.ndarray:
        black_white_image = []
        for row in image:
            black_white_row = []
            for pixel in row:
                black_white_pixel = int((pixel[0]*0.299 + pixel[1]*0.587 + pixel[2]*0.114)/3)
                black_white_row.append(black_white_pixel)
            black_white_image.append(black_white_row)
        return np.array(black_white_image, dtype=np.uint8)

    def __setImage(self, image : str | np.ndarray):
        raw_image = image
        if isinstance(image, str):
            if (not any([image.endswith(file_format) for file_format in ["jpeg", "jpg", "png"]])):
                raise ValueError("TomatoImage accepts JPEG, JPG and PNG file formats only")
            
            raw_image = cv2.imread(image)
            if raw_image is None:
                raise ValueError("The provided image path doesn't exists")
        
        self.__image = self.__black_white_image(raw_image)

    def getHeight(self) -> int:
        return len(self.__image[0])
    
    def getWidth(self) -> int:
        return len(self.__image)
    
    def getImageMatrix(self) -> np.ndarray:
        return self.__image

class Tomato:

    def __init__(self):
        pass

    def screen_shot_tomato(self) -> TomatoImage:

        monitor = mss().monitors[1]
        screenshot = mss().grab(monitor)
        screenshot_matrix = np.array(screenshot)
        screenshot_matrix_rgb = screenshot_matrix[:, :, :3]

        return TomatoImage(screenshot_matrix_rgb)

    @staticmethod
    def find_element_in_screen(element : TomatoImage, screen : TomatoImage, section : tuple[int, int, int, int] | None=None, under_step : int=1) -> tuple[int, int]:

        start_x, start_y, width, height = section if section is not None else (0, 0, screen.getWidth(), screen.getHeight())
        if under_step > element.getWidth() or under_step > element.getHeight(): raise ValueError("The parameter 'under_step' is larger than one of the element's dimensions")
        h_step = element.getWidth() // under_step
        v_step = element.getHeight()// under_step

        x_points = list(range(start_x, width - h_step + 1, h_step))
        if x_points[-1] + h_step < width:
            x_points.append(width - h_step)

        y_points = list(range(start_y, height - v_step + 1, v_step))
        if y_points[-1] + v_step < height:
            y_points.append(height - v_step)

        for i in x_points:
            for j in y_points:

                screen_section = screen.getImageMatrix()[i:i+h_step, j:j+v_step]
                cv2.imshow("Screen Section", screen_section)
                cv2.waitKey(0)


    def element_is_visible(self, element : TomatoImage, section : tuple[int, int, int, int] | None=None, under_step : int=1) -> bool:
        
        if isinstance(element, str): element = TomatoImage(element)

        screen = self.screen_shot_tomato()
        element_coordinates = self.find_element_in_screen(element, screen, section, under_step)

        return not any(coord is None for coord in element_coordinates)


