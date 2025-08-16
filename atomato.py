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

    def get_height(self) -> int:
        return len(self.__image)
    
    def get_width(self) -> int:
        return len(self.__image[0])
    
    def get_image_matrix(self) -> np.ndarray:
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

        mouse.move(0, 0)
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
            image = image.get_image_matrix()
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

        matrix1 = img1.get_image_matrix()
        matrix2 = img2.get_image_matrix()

        pixel_size_img1 = self.find_pixel_size(matrix1)
        pixel_size_img2 = self.find_pixel_size(matrix2)

        new_dim_matrix1 = (img1.get_width()//pixel_size_img1, img1.get_height()//pixel_size_img1)
        matrix1_redim = cv2.resize(matrix1, new_dim_matrix1)

        new_dim_matrix2 = (img2.get_width()//pixel_size_img2, img2.get_height()//pixel_size_img2)
        matrix2_redim = cv2.resize(matrix2, new_dim_matrix2)

        return TomatoImage(matrix1_redim), TomatoImage(matrix2_redim)

    def __safe_section(self, matrix : np.ndarray, i : int, j : int, image_height : int, image_width : int) -> np.ndarray:
        if i + image_height > matrix.shape[0] or j + image_width > matrix.shape[1]:
            return None
        return matrix[i:i+image_height, j:j+image_width]

    def __get_iteration_key_points(self, screen_dim : int, image_dim : int, step : int) -> list[int]:
        key_points = list(range(0, screen_dim - image_dim + 1, step))
        last_x = screen_dim - image_dim
        if key_points[-1] != last_x: key_points.append(last_x)
        return key_points

    def find_image_in_screen(self, image : TomatoImage, screen : TomatoImage, under_step : int | None=None, cutoff : float | None=None, show : bool=False) -> tuple[int, int]:

        start = time.time()

        image, screen = self.match_resolution(image, screen)
        image_matrix, screen_matrix = (image.get_image_matrix(), screen.get_image_matrix())
 
        if len(image_matrix) > len(screen_matrix) or len(image_matrix[0]) > len(screen_matrix[0]): raise ValueError("image must be smaller than screen in all dimensions")
        
        image_width, screen_width = (image.get_width(), screen.get_width())
        image_height, screen_height = (image.get_height(), screen.get_height())

        if under_step is None: under_step = min([image_width, image_height])
        if under_step > image_width or under_step > image_height: raise ValueError(f"The parameter 'under_step' is larger than one of the image's dimensions w:{image_width}, h:{image_height}")
        
        h_step = image_width // under_step
        v_step = image_height// under_step

        x_points = self.__get_iteration_key_points(screen_width, image_width, h_step)
        y_points = self.__get_iteration_key_points(screen_height, image_height, v_step)

        distances_centers_and_matrixes = []
        for i in x_points:
            for j in y_points:
                screen_section = self.__safe_section(screen_matrix, i, j, image_height, image_width)
                if screen_section is None: continue

                center_y = i + h_step // 2
                center_x = j + v_step // 2

                difference = self.distance_between_vectors(image_matrix, screen_section)

                if cutoff is not None and difference < cutoff:
                    if show:
                        end = time.time()
                        print(f"\nEXECUTION TIME (seconds): {end - start:.4f}\n")
                        cv2.imshow("image found", screen_section)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                    return (center_x, center_y)
                
                distances_centers_and_matrixes.append({"DIFFERENCE" : difference, "CENTER" : (center_x, center_y), "MATRIX" : screen_section})
        
        if cutoff is None:
            closest = min(distances_centers_and_matrixes, key=lambda x: x["DIFFERENCE"])

            if show:
                end = time.time()
                print(f"\nEXECUTION TIME (seconds): {end - start:.4f}\n")
                print(f"\nMIN CUTOFF: {closest['DIFFERENCE']}\n")

                cv2.imshow("image found", closest["MATRIX"])
                cv2.waitKey(0)
                cv2.destroyAllWindows()  

            return closest["CENTER"]
        return(None, None)

class TomatoElement:
    def __init__(self, image : TomatoImage | str | np.ndarray, x_pos : int | None=None, y_pos : int | None=None):
        self.__utils = TomatoUtils()
        self.__setImage(image)
        self.__height = self.__image.get_height()
        self.__width = self.__image.get_width()
        self.__x = x_pos
        self.__y = y_pos
    
    def __setImage(self, image : TomatoImage | str | np.ndarray):
        if not isinstance(image, TomatoImage): self.__image = TomatoImage(image)
        else: self.__image = image
    
    def get_image(self) -> TomatoImage:
        return self.__image
    
    def get_center(self) -> tuple:
        return (self.__x, self.__y)
    
    def get_width(self) -> int:
        return self.__width
    
    def get_height(self) -> int:
        return self.__height

    def move_center(self, x_delta : int=0, y_delta : int=0):
        try:
            x_delta = int(x_delta)
            y_delta = int(y_delta)
        except ValueError as e:
            raise TypeError(
                f"Invalid input: x_delta='{x_delta}', y_delta='{y_delta}'. "
                "Both must be convertible to integers."
            ) from e

        self.__x += x_delta
        self.__y += y_delta

    def __wait_until_mouse_hoover(self, timeout : int=10):
        counter = 0
        while mouse.get_position() != self.get_center():
            time.sleep(0.1)
            counter += 1
            if counter >= timeout: break
            continue

    def update_coordinates(self, under_step : int | None=None, cutoff : float | None=None, show : bool=False):
        screen = self.__utils.screen_shot_tomato()
        self.__x, self.__y = self.__utils.find_image_in_screen(self.get_image(), screen, under_step, cutoff, show)
        return self

    def click(self, button : str="L"):
        if None in self.get_center(): raise AttributeError("Cannot click: element coordinates are not set.")
        mouse.move(self.__x, self.__y)
        self.__wait_until_mouse_hoover()
        if button.upper() == "L": mouse.click("left")
        elif button.upper() == "R": mouse.click("right")
        else: raise ValueError("Attribute 'button' must be either 'L' or 'R', referring to the left and right mouse buttons, respectively.")

    def type(self, text : str):
        self.click()
        keyboard.write(text)

    def input(self, info : str):
        self.type(info)
        keyboard.press_and_release("enter")

    def delete_input(self):
        self.click()
        self.click()
        keyboard.press_and_release("backspace")

    def wait_visibility(self, under_step: int | None = None, cutoff: float | None = None, timeout: int = 10, wait_invisibility: bool = False):
        start = time.time()

        while True:
            screen = self.__utils.screen_shot_tomato()
            coord = self.__utils.find_image_in_screen(self.get_image(), screen, under_step, cutoff)

            duration = time.time() - start
            if duration > timeout:
                state = "invisible" if wait_invisibility else "visible"
                raise TimeoutError(f"Element wasn't {state} before timeout")

            if (wait_invisibility and None in coord) or (not wait_invisibility and None not in coord):
                break

        self.__x, self.__y = coord if coord else (None, None)
        return self




