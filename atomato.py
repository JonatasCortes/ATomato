import numpy as np
import keyboard
import mouse
import cv2
from mss import mss
import os
import time
from imagehash import average_hash
from scipy.signal import find_peaks
from PIL import Image

class TomatoImage:
    def __init__(self, image : str | np.ndarray):
        self.__setImage(image)

    def __black_white_image(self, image : np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        return image

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
    
    def getImageMatrix(self) -> np.ndarray:
        return self.__image

class Tomato:

    def __init__(self):
        pass

    def __safe_section(self, matrix : np.ndarray, i : int, j : int, height : int, width : int) -> np.ndarray:
        if i + height > matrix.shape[0] or j + width > matrix.shape[1]:
            return None
        return matrix[i:i+height, j:j+width]


    def distance_between_vectors(self, v1 : np.ndarray, v2 : np.ndarray) -> float:
        if v1.shape != v2.shape: raise ValueError("vectors must have the exact same dimensions")
        v1 = v1.astype(np.float32) / 255.0
        v2 = v2.astype(np.float32) / 255.0
        return np.linalg.norm(v1 - v2)

    def screen_shot_tomato(self) -> TomatoImage:

        monitor = mss().monitors[1]
        screenshot = mss().grab(monitor)
        screenshot_matrix = np.array(screenshot)
        screenshot_matrix_rgb = screenshot_matrix[:, :, :3]

        return TomatoImage(screenshot_matrix_rgb)

    def find_pixel_size(self, image: np.ndarray | TomatoImage | str, prominence: float = 10, tolerance: float = 0.3) -> int:
        if isinstance(image, str):
            image = TomatoImage(image)
        if isinstance(image, TomatoImage):
            matrix = image.getImageMatrix()
        if isinstance(image, np.ndarray):
            matrix = image

        def analyze_direction(gray: np.ndarray, axis: int) -> int:
            dx, dy = (1, 0) if axis == 0 else (0, 1)
            sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=3)
            profile = np.mean(np.abs(sobel), axis=axis)
            peaks, _ = find_peaks(profile, prominence=prominence)

            if len(peaks) >= 3:
                distances = np.diff(peaks)
                median = np.median(distances)
                std_dev = np.std(distances)
                # Verifica se os blocos são regulares
                if std_dev / median < tolerance:
                    return int(round(median))
            return 0

        # Converte para escala de cinza se necessário
        if len(matrix.shape) == 3:
            matrix = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)

        # Analisa horizontal (eixo 0) e vertical (eixo 1)
        horizontal_size = analyze_direction(matrix, axis=0)
        vertical_size = analyze_direction(matrix, axis=1)

        # Combina os resultados
        if horizontal_size and vertical_size:
            return int(round((horizontal_size + vertical_size) / 2))
        elif horizontal_size:
            return horizontal_size
        elif vertical_size:
            return vertical_size
        else:
            return 1  # Não detectável ou não confiável

    def match_resolution(self, img1: np.ndarray | TomatoImage | str, img2: np.ndarray | TomatoImage | str, downscale_factor: int = 2, min_size: int = 32) -> tuple[np.ndarray, np.ndarray]:

        if isinstance(img1, str) or isinstance(img1, np.ndarray): img1 = TomatoImage(img1)
        if isinstance(img2, str) or isinstance(img2, np.ndarray): img2 = TomatoImage(img2)

        matrix1 = img1.getImageMatrix()
        matrix2 = img2.getImageMatrix()

        pixel_size_img1 = self.find_pixel_size(matrix1)
        pixel_size_img2 = self.find_pixel_size(matrix2)

        new_dim_matrix1 = (img1.getWidth()//pixel_size_img1, img1.getHeight()//pixel_size_img1)
        matrix1_redim = cv2.resize(matrix1, new_dim_matrix1)

        new_dim_matrix2 = (img2.getWidth()//pixel_size_img2, img2.getHeight()//pixel_size_img2)
        matrix2_redim = cv2.resize(matrix2, new_dim_matrix2)

        return TomatoImage(matrix1_redim), TomatoImage(matrix2_redim)

    def find_element_in_screen(self, element : TomatoImage, screen : TomatoImage, section : tuple[int, int, int, int] | None=None, under_step : int | None=None, cutoff : float=8, debug : bool=False, find_closest : bool=False) -> tuple[int, int]:

        element, screen = self.match_resolution(element, screen)

        element_width = element.getWidth()
        element_height = element.getHeight()

        if under_step is None: under_step = min([element_width, element_height])//2

        start_x, start_y, width, height = section if section is not None else (0, 0, screen.getWidth(), screen.getHeight())
        if under_step > element_width or under_step > element_height: raise ValueError(f"The parameter 'under_step' is larger than one of the element's dimensions w:{element_width}, h:{element_height}")
        h_step = element_width // under_step
        v_step = element_height// under_step

        x_points = list(range(start_x, width - element_width + 1, h_step))
        last_x = width - element_width
        if x_points[-1] != last_x: x_points.append(last_x)

        y_points = list(range(start_y, height - element_height + 1, v_step))
        last_y = height - element_height
        if y_points[-1] != last_y: y_points.append(last_y)

        distances_and_centers = []
        counter = 0
        for i in x_points:
            for j in y_points:

                screen_section = self.__safe_section(screen.getImageMatrix(), i, j, element_height, element_width)
                if screen_section is None: continue

                center_x = i + h_step // 2
                center_y = j + v_step // 2

                difference = self.distance_between_vectors(element.getImageMatrix(), screen_section)

                if difference < cutoff:
                    return (center_x, center_y)
                
                distances_and_centers.append((difference, (center_x, center_y)))
                counter += 1
        
        if find_closest:
            min_difference = min(distances_and_centers, key=lambda x: x[0])[0]
            closest_coord = min(distances_and_centers, key=lambda x: x[0])[1]
            print(min_difference)
            return closest_coord

    def element_is_visible(self, element : TomatoImage, section : tuple[int, int, int, int] | None=None, under_step : int=1) -> bool:
        
        if isinstance(element, str): element = TomatoImage(element)

        screen = self.screen_shot_tomato()
        element_coordinates = self.find_element_in_screen(element, screen, section, under_step)

        return not any(coord is None for coord in element_coordinates)


