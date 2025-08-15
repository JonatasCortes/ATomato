import numpy as np
import keyboard
import mouse
import cv2
from mss import mss
import os
import time
from scipy.signal import find_peaks

class TomatoImage:
    def __init__(self, image : str | np.ndarray):
        self.__setImage(image)

    def __black_white_image(self, image : np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        return image

    def __setImage(self, image : str | np.ndarray):
        if not isinstance(image, (str, np.ndarray)): raise TypeError("Image must be a file path (jpeg, jpg, png) or an instance of np.ndarray")
        raw_image = image
        if isinstance(image, str):
            if (not any([image.endswith(file_format) for file_format in ["jpeg", "jpg", "png"]])):
                raise ValueError("TomatoImage accepts JPEG, JPG and PNG file formats only")
            
            raw_image = cv2.imread(image)
            if raw_image is None:
                raise ValueError("The provided image path doesn't exists")
        
        if raw_image.shape == 0: raise ValueError("The given image could not be resolved")
        self.__image = self.__black_white_image(raw_image)

    def getHeight(self) -> int:
        return len(self.__image)
    
    def getWidth(self) -> int:
        return len(self.__image[0])
    
    def getImageMatrix(self) -> np.ndarray:
        return self.__image

class TomatoUtils:

    def __init__(self):
        pass

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

    def __pixel_size_by_axis(self, image: np.ndarray, axis: int, prominence : float, tolerance : float) -> int:
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        dx, dy = (1, 0) if axis == 0 else (0, 1)
        sobel = cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize=3)
        profile = np.mean(np.abs(sobel), axis=axis)
        peaks, _ = find_peaks(profile, prominence=prominence)

        if len(peaks) >= 10:
            distances = np.diff(peaks)
            median = np.median(distances)
            std_dev = np.std(distances)
            # Verifica se os blocos são regulares
            if std_dev / median < tolerance:
                return int(round(median))
        return 0

    def find_pixel_size(self, image: np.ndarray | TomatoImage | str, prominence: float = 10, tolerance: float = 0.3) -> int:
        if isinstance(image, str):
            image = TomatoImage(image)
        if isinstance(image, TomatoImage):
            image = image.getImageMatrix()
        if not isinstance(image, np.ndarray):
            raise TypeError("Image type is not suported")
        
        matrix = image

        horizontal_size = self.__pixel_size_by_axis(matrix, axis=0, prominence=prominence, tolerance=tolerance)
        vertical_size = self.__pixel_size_by_axis(matrix, axis=1, prominence=prominence, tolerance=tolerance)

        if horizontal_size and vertical_size:
            return int(round((horizontal_size + vertical_size) / 2))
        elif horizontal_size:
            return horizontal_size
        elif vertical_size:
            return vertical_size
        else:
            return 1  # Não detectável ou não confiável

    def match_resolution(self, img1: np.ndarray | TomatoImage | str, img2: np.ndarray | TomatoImage | str) -> tuple[np.ndarray, np.ndarray]:

        if not isinstance(img1, TomatoImage): img1 = TomatoImage(img1)
        if not isinstance(img2, TomatoImage): img2 = TomatoImage(img2)

        matrix1 = img1.getImageMatrix()
        matrix2 = img2.getImageMatrix()

        pixel_size_img1 = self.find_pixel_size(matrix1)
        pixel_size_img2 = self.find_pixel_size(matrix2)

        new_dim_matrix1 = (img1.getWidth()//pixel_size_img1, img1.getHeight()//pixel_size_img1)
        matrix1_redim = cv2.resize(matrix1, new_dim_matrix1)

        new_dim_matrix2 = (img2.getWidth()//pixel_size_img2, img2.getHeight()//pixel_size_img2)
        matrix2_redim = cv2.resize(matrix2, new_dim_matrix2)

        return TomatoImage(matrix1_redim), TomatoImage(matrix2_redim)

class Tomato:

    def __init__(self):
        self.__utils = TomatoUtils()

    def __safe_section(self, matrix : np.ndarray, i : int, j : int, element_height : int, element_width : int) -> np.ndarray:
        if i + element_height > matrix.shape[0] or j + element_width > matrix.shape[1]:
            return None
        return matrix[i:i+element_height, j:j+element_width]

    def __get_iteration_key_points(self, screen_dim : int, element_dim : int, step : int) -> list[int]:
        key_points = list(range(0, screen_dim - element_dim + 1, step))
        last_x = screen_dim - element_dim
        if key_points[-1] != last_x: key_points.append(last_x)
        return key_points

    def find_element_in_screen(self, element : TomatoImage, screen : TomatoImage, under_step : int | None=None, cutoff : float | None=None, show : bool=False) -> tuple[int, int]:

        element, screen = self.__utils.match_resolution(element, screen)
        element_matrix, screen_matrix = (element.getImageMatrix(), screen.getImageMatrix())
 
        if len(element_matrix) > len(screen_matrix) or len(element_matrix[0]) > len(screen_matrix[0]): raise ValueError("Element must be smaller than screen in all dimensions")
        
        element_width, screen_width = (element.getWidth(), screen.getWidth())
        element_height, screen_height = (element.getHeight(), screen.getHeight())

        if under_step is None: under_step = min([element_width, element_height])
        if under_step > element_width or under_step > element_height: raise ValueError(f"The parameter 'under_step' is larger than one of the element's dimensions w:{element_width}, h:{element_height}")
        
        h_step = element_width // under_step
        v_step = element_height// under_step

        x_points = self.__get_iteration_key_points(screen_width, element_width, h_step)
        y_points = self.__get_iteration_key_points(screen_height, element_height, v_step)

        distances_centers_and_matrixes = []

        for i in x_points:
            for j in y_points:

                screen_section = self.__safe_section(screen_matrix, i, j, element_height, element_width)
                if screen_section is None: continue

                center_x = i + h_step // 2
                center_y = j + v_step // 2

                difference = self.__utils.distance_between_vectors(element_matrix, screen_section)

                if cutoff is not None and difference < cutoff:
                    if show:
                        cv2.imshow("element found", screen_section)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                    return (center_x, center_y)
                
                distances_centers_and_matrixes.append({"DIFFERENCE" : difference, "CENTER" : (center_x, center_y), "MATRIX" : screen_section})
        
        if cutoff is None:
            closest = min(distances_centers_and_matrixes, key=lambda x: x["DIFFERENCE"])

            if show:
                cv2.imshow("element found", closest["MATRIX"])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                print(f"\nMIN CUTOFF: {closest['DIFFERENCE']}\n")

            return closest["CENTER"]

    def element_is_visible(self, element : TomatoImage, section : tuple[int, int, int, int] | None=None, under_step : int=1) -> bool:
        
        if isinstance(element, str): element = TomatoImage(element)

        screen = self.screen_shot_tomato()
        element_coordinates = self.find_element_in_screen(element, screen, section, under_step)

        return not any(coord is None for coord in element_coordinates)


