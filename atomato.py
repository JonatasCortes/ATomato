import numpy as np
import keyboard
import mouse
import cv2
import mss # APRENDER A USAR
import os

class Rect:
    def __init__(self, x : int | None=0, y : int | None=0, width : int | None=0, height : int | None=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

class TomatoImage:
    def __init__(self, image_path : str):
        self.__setImage(image_path)

    def __black_white_image(self, image : np.ndarray) -> np.ndarray:
        black_white_image = []
        for row in image:
            black_white_row = []
            for pixel in row:
                black_white_pixel = int((pixel[0]*0.299 + pixel[1]*0.587 + pixel[2]*0.114)/3)
                black_white_row.append(black_white_pixel)
            black_white_image.append(black_white_row)
        return np.array(black_white_image, dtype=np.uint8)

    def __setImage(self, image_path : str):
        if not any([image_path.endswith(file_format) for file_format in ["jpeg", "jpg", "png"]]):
            raise ValueError("TomatoImage accepts JPEG, JPG and PNG file formats only")
        
        raw_image = cv2.imread(image_path)
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
    def element_is_visible(self, tomato : TomatoImage | str):
        pass