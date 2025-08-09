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

    def screen_shot_tomato(self) -> TomatoImage:

        monitor = mss().monitors[1]
        screenshot = mss().grab(monitor)
        screenshot_matrix = np.array(screenshot)
        screenshot_matrix_rgb = screenshot_matrix[:, :, :3]

        return TomatoImage(screenshot_matrix_rgb)

    @staticmethod
    def find_pixel_size(image: np.ndarray | TomatoImage | str, prominence: float = 10, tolerance: float = 0.3) -> int:
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

    @staticmethod
    def match_resolution(img1: np.ndarray | TomatoImage | str, img2: np.ndarray | TomatoImage | str, downscale_factor: int = 2, min_size: int = 32) -> tuple[np.ndarray, np.ndarray]:

        if isinstance(img1, str) or isinstance(img1, np.ndarray): img1 = TomatoImage(img1)
        if isinstance(img2, str) or isinstance(img2, np.ndarray): img2 = TomatoImage(img2)

        matrix1 = img1.getImageMatrix()
        matrix2 = img2.getImageMatrix()

        pixel_size_img1 = Tomato.find_pixel_size(matrix1)
        pixel_size_img2 = Tomato.find_pixel_size(matrix2)

        new_dim_matrix1 = (img1.getWidth()//pixel_size_img1, img1.getHeight()//pixel_size_img1)
        matrix1_redim = cv2.resize(matrix1, new_dim_matrix1)

        new_dim_matrix2 = (img2.getWidth()//pixel_size_img2, img2.getHeight()//pixel_size_img2)
        matrix2_redim = cv2.resize(matrix2, new_dim_matrix2)

        return TomatoImage(matrix1_redim), TomatoImage(matrix2_redim)

    @staticmethod
    def find_element_in_screen(element : TomatoImage, screen : TomatoImage, section : tuple[int, int, int, int] | None=None, under_step : int=1, cutoff : int=5, debug : bool=False) -> tuple[int, int]:

        element, screen = Tomato.match_resolution(element, screen)

        element_width = element.getWidth()
        element_height = element.getHeight()

        start_x, start_y, width, height = section if section is not None else (0, 0, screen.getWidth(), screen.getHeight())
        if under_step > element_width or under_step > element_height: raise ValueError("The parameter 'under_step' is larger than one of the element's dimensions")
        h_step = element_width // under_step
        v_step = element_height// under_step

        x_points = list(range(start_x, width - h_step + 1, h_step))
        if x_points[-1] + h_step < width:
            x_points.append(width - h_step)

        y_points = list(range(start_y, height - v_step + 1, v_step))
        if y_points[-1] + v_step < height:
            y_points.append(height - v_step)

        element_hash = average_hash(Image.fromarray(element.getImageMatrix()))

        counter = 0
        for i in x_points:
            for j in y_points:

                screen_section = screen.getImageMatrix()[i:i+element_height, j:j+element_width]
                section_hash = average_hash(Image.fromarray(screen_section))

                center_x = i + h_step // 2
                center_y = j + v_step // 2

                if debug:
                    cv2.imshow("element", element.getImageMatrix())
                    cv2.imshow("screen section", screen_section)
                    time.sleep(1)
                    cv2.destroyAllWindows()
                    print(np.abs(element_hash - section_hash))

                if np.abs(element_hash - section_hash) < cutoff:
                    return (center_x, center_y)
                
                counter += 1

    def element_is_visible(self, element : TomatoImage, section : tuple[int, int, int, int] | None=None, under_step : int=1) -> bool:
        
        if isinstance(element, str): element = TomatoImage(element)

        screen = self.screen_shot_tomato()
        element_coordinates = self.find_element_in_screen(element, screen, section, under_step)

        return not any(coord is None for coord in element_coordinates)


