import numpy as np
import keyboard
import mouse
import cv2
from mss import mss
import os

class Rect:
    def __init__(self, x : int | None=0, y : int | None=0, width : int | None=0, height : int | None=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

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
        return len(self.__image)
    
    def getWidth(self) -> int:
        return len(self.__image[0])
    
    def getImageMatrix(self) -> list[list[int]]:
        return self.__image

class Tomato:

    def __screen_shot_tomato(self) -> TomatoImage:

        monitor = mss().monitors[1]
        screenshot = mss().grab(monitor)
        screenshot_matrix = np.array(screenshot)
        screenshot_matrix_rgb = screenshot_matrix[:, :, :3]

        return TomatoImage(screenshot_matrix_rgb)

    def find_tomato_in_screen(self, tomato : TomatoImage, under_step : int=1) -> tuple[int, int]:
        
        screen = self.__screen_shot_tomato()

        if under_step > tomato.getWidth() or under_step > tomato.getHeight(): raise ValueError("parameter 'under_step' is larger than one of tomato dimensions")
        h_step = tomato.getWidth() // under_step
        v_step = tomato.getHeight()// under_step

        for i in range(0, screen.getWidth(), h_step):
            for j in range(0, screen.getHeight(), v_step):
                pass

    def element_is_visible(self, element : TomatoImage | str, step : int | None=None) -> bool:
        
        if isinstance(element, str): element = TomatoImage(element)

        element_coordinates = self.find_tomato_in_screen(element)

        return not any(coord is None for coord in element_coordinates)


