import numpy as np
import keyboard
import mouse
import cv2
from mss import mss
import os
import time
from scipy.signal import find_peaks, fftconvolve
import webbrowser
import pygetwindow as gw
from pywinauto import Desktop
import operator
from typing import overload, Union
from scipy.ndimage import convolve


class TomatoImage:
    def __init__(self, image : str | np.ndarray):
        self.__utils = TomatoUtils()
        self.__set_grayscale_image(image)

    def __eq__(self, other : 'TomatoImage'):
        if not isinstance(other, TomatoImage): return False
        try:
            tolerance = 10
            difference = np.mean(np.abs(self.get_image_matrix() - other.get_image_matrix()))
            if difference > tolerance: return False
        except:
            return False
        return True

    def __getitem__(self, key : int | slice | tuple) -> "TomatoImage":
        if isinstance(key, slice):
            return TomatoImage(self.__image[key])
        elif isinstance(key, tuple) and len(key)==2:
            y_slice, x_slice = key
            if not all([isinstance(y_slice, slice), isinstance(x_slice, slice)]): 
                raise KeyError("Invalid slicing.")
            return TomatoImage(self.__image[y_slice,x_slice])
        else:
            try:
                index = operator.index(key)
                return TomatoImage(self.__image[index])
            except Exception as e:
                raise KeyError(f"Invalid index: {e}")

    def __set_grayscale_image(self, image : str | np.ndarray):
        """
        Validates and converts the input image to grayscale, then stores it in self.__image.

        Accepts either a NumPy array (RGB or RGBA) or a file path to an image in JPEG, JPG, or PNG format.
        If a file path is provided, the image is read using OpenCV. The image is then converted to grayscale
        using perceptual luminance weights and stored internally.

        Args:
            image (str | np.ndarray): Input image as a NumPy array or a file path to a JPEG, JPG, or PNG image.

        Raises:
            TypeError: If the input is neither a string nor a NumPy array.
            ValueError: If the file format is unsupported, the path does not exist, or the image cannot be resolved.

        Returns:
            None
        """

        if not isinstance(image, (str, np.ndarray)): raise TypeError("The provided image is not a file path (jpeg, jpg, png) neither an instance of np.ndarray")
        raw_image = image
        if isinstance(image, str):
            if (not any([image.endswith(file_format) for file_format in ["jpeg", "jpg", "png"]])):
                raise ValueError(f"{image} is not a file path (jpeg, jpg, png)")
            
            raw_image = cv2.imread(image)
            if raw_image is None:
                raise ValueError(f"There is no image in {image}")
        
        if raw_image.shape == 0: raise ValueError("The provided image could not be resolved")
        self.__image = self.__utils.grayscale_image(raw_image)

    def show(self):
        cv2.imshow("Image", self.__image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()   

    def get_height(self) -> int:
        return int(len(self.__image))
    
    def get_width(self) -> int:
        try:
            return int(len(self.__image[0]))
        except:
            return 1
    
    def get_image_matrix(self) -> np.ndarray:
        return self.__image

class TomatoScreen:
    def __init__(self, screen : TomatoImage | np.ndarray | str):
        self.__utils = TomatoUtils()
        if not isinstance(screen, TomatoImage): screen = TomatoImage(screen)
        self.__image = screen
        self.__set_elements(self.__image)

    def __eq__(self, other) -> bool:
        if not isinstance(other, TomatoScreen): return False

        self_elements, other_elements = self.list_elements(), other.list_elements()
        if len(self_elements) != len(other_elements): return False

        for self_element, other_element in zip(self_elements, other_elements):
            if self_element != other_element:
                return False
        
        return True

    def __getitem__(self, key : slice | tuple) -> "TomatoScreen":
        if isinstance(key, slice):
            sliced = TomatoScreen(self.__image[key])
            sliced.move_centers(y_delta=key.start)
            return sliced
        elif isinstance(key, tuple) and len(key)==2:
            y_slice, x_slice = key
            if not all([isinstance(y_slice, slice), isinstance(x_slice, slice)]): 
                raise KeyError("Invalid slicing operator.")
            sliced = TomatoScreen(self.__image[key])
            sliced.move_centers(x_delta=x_slice.start, y_delta=y_slice.start)
            return sliced
        else:
            raise KeyError("Invalid slicing. Key must be either a slice instance or a tuple of slices")

    def get_screen_matrix(self) -> np.ndarray:
        return self.__image.get_image_matrix()
    
    def get_screen_image(self) -> TomatoImage:
        return self.__image

    def get_height(self) -> int:
        return self.__image.get_height()

    def get_width(self) -> int:
        return self.__image.get_width()

    def __set_elements(self, screen : TomatoImage):
        screen_mat = screen.get_image_matrix()
        edges = self.__utils.edge_detection(screen_mat)
        edges_mat = edges.get_image_matrix()
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(edges_mat, connectivity=8)

        self.__elements = []

        for i in range(num_labels):

            cv2_stats = [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]
            x, y, w, h = [stats[i, stat] for stat in cv2_stats]

            center_x = x + w//2
            center_y = y + h//2

            element = screen_mat[y:y+h, x:x+w]
            if len(element) > 10 and len(element[0]) > 10:
                self.__elements.append(TomatoElementFromScreen(element, int(center_x), int(center_y)))

    def list_elements(self) -> list['TomatoElement']:
        return self.__elements

    def get_element_by_id(self, id : int) -> 'TomatoElement':
        return self.__elements[id]

    def find_element(self, target : 'TomatoElement'):

        for element in self.list_elements():
            if element.get_image() == target.get_image():
                return element
        
        raise LookupError("The provided element could not be located.")

    def move_centers(self, x_delta : int=0, y_delta : int=0, start_index : int=0, stop_index : int | None=None, step : int=1):
        if stop_index is None: stop_index = len(self.__elements)
        for element in self.list_elements()[start_index:stop_index:step]:
            element.move_center(x_delta, y_delta)

    def show_elements(self):
        for index, element in enumerate(self.list_elements()):
            print(f"Index: {index}")
            element.show()

    def show(self):
        self.__image.show()

class TomatoWindow:
    def __init__(self, url : str, window_name : str | None=None):
        self.__utils = TomatoUtils()
        self.__url = url
        self.__name = window_name

    def __set_name(self, url : str):
        url_components = url.split("\\")
        if len(url_components)==1: url_components = url.split("/")
        url_tail = url_components[-1].upper().replace(" ", "")

        for candidate_title in gw.getAllTitles():
            normalized_candidate_title = candidate_title.replace(" ", "").upper()
            if normalized_candidate_title.endswith("GOOGLECHROME") and url_tail in normalized_candidate_title:
                self.__name = candidate_title
                return
        raise ValueError("Unable to infer 'window_name' from the provided URL.\nPlease specify the 'window_name' parameter when initializing TomatoWindow.")

    def get_name(self) -> str:
        return self.__name

    def __set_window(self):
        self.__window = Desktop(backend="uia").window(title=self.__name)
        rect = self.__window.rectangle()
        self.__width = rect.width()
        self.__height = rect.height()
        self.__left = rect.left
        self.__top = rect.top

        self.__monitor = {"top": self.__top, "left": self.__left, "width": self.__width, "height": self.__height}

    def screen_shot(self) -> TomatoScreen:
        screen_shot = mss().grab(self.__monitor)
        screenshot_matrix = np.array(screen_shot)
        screenshot_matrix_rgb = screenshot_matrix[:, :, :3]
        return TomatoScreen(screenshot_matrix_rgb)

    def open(self, timeout : float=10):
        webbrowser.open_new(self.__url)

        if self.__name:
            self.__set_window()
            return self.__window.set_focus()

        end_time = time.time() + timeout
        while time.time() < end_time:
            try:
                self.__set_name(self.__url)
                return self.__set_window()
            except:
                continue
        raise TimeoutError("Unable to infer 'window_name' from the provided URL before Timeout.\nPlease specify the 'window_name' parameter when initializing TomatoWindow, or raise the timeout.")

    def close(self):
        keyboard.press("alt")
        keyboard.press("f4")
        keyboard.release()

    def wait_change(self, timeout : float=10):
        time.sleep(0.05)
        end_time = time.time() + timeout - 0.25
        screen = self.screen_shot()
        while time.time() < end_time:
            new_screen = self.screen_shot()
            if screen != new_screen:
                return
        raise TimeoutError("Window didn't changed before timeout.")

    def wait_stability(self, stable_time : float=2, timeout : float=10):
        end_time = time.time() + timeout
        while time.time() < end_time:
            try:
                self.wait_change(stable_time)
            except:
                return
        raise TimeoutError("Window wasn't stable before timeout")

class TomatoElement:
    def __init__(self, image : TomatoImage | str | np.ndarray, x_pos : int | None=None, y_pos : int | None=None):
        self._utils = TomatoUtils()
        image = image if isinstance(image, TomatoImage) else TomatoImage(image)
        self._set_status(image, x_pos, y_pos)
    
    def __eq__(self, other : 'TomatoElement'):
        if not isinstance(other, TomatoElement): return False

        if self.get_center() != other.get_center(): return False
        if not self.get_image() == other.get_image(): return False

        return True

    def __getitem__(self, key : int | slice | tuple[slice, slice]) -> "TomatoElement":
        if isinstance(key, slice):
            start = key.start
            stop = key.stop
            step = key.step
            return TomatoElement(self._image[start:stop:step])
        elif isinstance(key, tuple) and len(key)==2:
            y_slice, x_slice = key
            if not all([isinstance(y_slice, slice), isinstance(x_slice, slice)]): 
                raise KeyError("Invalid slicing.")
            return TomatoElement(self._image[key])
        else:
            try:
                index = operator.index(key)
                return TomatoElement(self._image[index])
            except Exception as e:
                raise KeyError(f"Invalid index: {e}")
    
    def _set_status(self, image : TomatoImage, x_pos : int | None=None, y_pos : int | None=None):
        image_mat = image.get_image_matrix()
        edges = self._utils.edge_detection(image)
        edges_mat = edges.get_image_matrix()
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(edges_mat, connectivity=8)

        if num_labels > 2:  raise ValueError("The input image appears to contain multiple distinct elements.\nIf this is not the case, please ensure that there is no significant noise or artifacts affecting the segmentation.")

        cv2_stats = [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]
        x, y, w, h = [stats[1, stat] for stat in cv2_stats]

        self._width = w
        self._height = h
        self._x = x_pos if x_pos else None
        self._y = y_pos if y_pos else None

        formated_image = image_mat[y:y+h, x:x+w]
        self._image = TomatoImage(formated_image)

    def get_image(self) -> TomatoImage:
        return self._image
    
    def get_image_matrix(self) -> np.ndarray:
        return self._image.get_image_matrix()

    def get_center(self) -> tuple:
        return (self._x, self._y)
    
    def get_width(self) -> int:
        return self._width
    
    def get_height(self) -> int:
        return self._height

    def move_center(self, x_delta : int=0, y_delta : int=0):
        try:
            x_delta = int(x_delta)
            y_delta = int(y_delta)
        except ValueError as e:
            raise TypeError(
                f"Invalid input: x_delta='{x_delta}', y_delta='{y_delta}'. "
                "Both must be convertible to integers."
            ) from e

        self._x += x_delta
        self._y += y_delta

    def _wait_until_mouse_hoover(self, timeout : int=10):
        counter = 0
        while mouse.get_position() != self.get_center():
            time.sleep(0.1)
            counter += 1
            if counter >= timeout: break
            continue

    def click(self, button : str="L"):
        if None in self.get_center(): raise AttributeError("Cannot click: element coordinates are not set.")
        mouse.move(self._x, self._y)
        self._wait_until_mouse_hoover()
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
        for _ in range(3):
            self.click()
        keyboard.press_and_release("backspace")

    def show(self):
        self._image.show()

class TomatoElementFromScreen(TomatoElement):
    def __init__(self, image : TomatoImage | str | np.ndarray, x_pos : int | None=None, y_pos : int | None=None):
        self._image = image if isinstance(image, TomatoImage) else TomatoImage(image)
        self._height = self._image.get_height()
        self._width = self._image.get_width()
        self.__set_x(x_pos)
        self.__set_y(y_pos)
    
    def __eq__(self, other : 'TomatoElement'):
        return super().__eq__(other)

    def __getitem__(self, key : int | slice | tuple[slice, slice]):
        return super().__getitem__(key)

    def __set_x(self, x_pos : int):
        try:
            self._x = int(x_pos)
        except:
            raise TypeError("Parameter 'x_pos' must be convertable to integer")

    def __set_y(self, y_pos : int):
        try:
            self._y = int(y_pos)
        except:
            raise TypeError("Parameter 'y_pos' must be convertable to integer")

class TomatoUtils:
    def __init__(self):
        pass

    def noise_reduction(self, image : np.ndarray) -> np.ndarray:
        connectivity = 8

        output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

        num_stats = output[0]
        labels = output[1]
        stats = output[2]

        new_image = image.copy()

        for label in range(num_stats):
            if stats[label,cv2.CC_STAT_AREA] < 170:
                new_image[labels == label] = 0

        return new_image

    def edge_detection(self, image : TomatoImage | np.ndarray) -> TomatoImage:
        if isinstance(image, TomatoImage): image = image.get_image_matrix()
        if not isinstance(image, np.ndarray): raise TypeError("Image must be an instance of TomatoImage or np.ndarray")

        edges = cv2.GaussianBlur(image, (5,5), 0) - cv2.GaussianBlur(image, (5,5), 10)
        cleaned = self.noise_reduction(edges)
        
        return TomatoImage(cleaned)

    def grayscale_image(self, image : np.ndarray) -> np.ndarray:
        """
        Converts a colored image to its black-and-white (grayscale) version.

        If the input image is already in grayscale format, it is returned unchanged.
        Otherwise, the method computes luminance using perceptual weights for the
        red, green, and blue channels: [0.299, 0.587, 0.114].

        Args:
            image (np.ndarray): Input image as a NumPy array in grayscale, RGB, or RGBA format.

        Raises:
            TypeError: If the input is not a NumPy array.

        Returns:
            np.ndarray: Grayscale image with dtype np.uint8 and shape (H, W).
        """

        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be a NumPy ndarray.")
        if len(image.shape) == 3:
            return np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        return image

    def grayscale_image_distance(self, img1 : np.ndarray | TomatoImage, img2 : np.ndarray | TomatoImage) -> float:
        """
        Returns the Euclidean distance between two grayscaled images (np.ndarray) or TomatoImage instances.

        If either of the provided images is not in grayscale or not an instance of TomatoImage, it will be converted to prevent errors.

        Args:
            img1 (np.ndarray | TomatoImage): First input image.
            img2 (np.ndarray | TomatoImage): Second input image.

        Raises:
            TypeError: If either of the provided images is neither an instance of np.ndarray nor TomatoImage.
            ValueError: If the provided images have different shapes.

        Returns:
            float: The Euclidean distance between the grayscaled versions of the provided images.
        """

        if not all([isinstance(v, (np.ndarray, TomatoImage)) for v in (img1,img2)]): raise TypeError("One of the provided vectors is not an instance of np.ndarray")
        
        img1, img2 = [image.get_image_matrix() if isinstance(image, TomatoImage) else image for image in (img1, img2)]      #Extract img1 and img2 matrixes if needed
        img1, img2 = [self.grayscale_image(image) if image.ndim == 3 and image.shape[2] > 1 else image for image in (img1, img2)]     #Convert img1 and img2 to grayscale if needed
        if img1.shape != img2.shape: raise ValueError("The provided vectors have incompatible shapes.")
        
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

        return np.linalg.norm(img1 - img2)

    def __pixel_size_by_axis(self, image: np.ndarray, axis: int, prominence : float, tolerance : float) -> int:
        
        if len(image.shape) == 3:
            image = self.grayscale_image(image)

        dx, dy = (1, 0) if axis == 0 else (0, 1)
        sobel = cv2.Scharr(image, cv2.CV_64F, dx, dy)
        profile = np.mean(np.abs(sobel), axis=axis)
        peaks, _ = find_peaks(profile, prominence=prominence)

        if len(peaks) >= 15:
            distances = np.diff(peaks)
            median = np.median(distances)
            std_dev = np.std(distances)
            # Verifica se os blocos são regulares
            if std_dev / median < tolerance:
                return int(round(median))
        return 0

    def find_pixel_size(self, image: np.ndarray | TomatoImage | str, prominence: float = 10, tolerance: float = 0.2) -> int:
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

    def __safe_section(self, screen : np.ndarray, outher_loop_index : int, inner_loop_index : int, target_height : int, target_width : int) -> np.ndarray:
        if outher_loop_index + target_height > screen.shape[0] or inner_loop_index + target_width > screen.shape[1]:
            return None
        return screen[outher_loop_index:outher_loop_index+target_height, inner_loop_index:inner_loop_index+target_width]

    def find_element_in_screen(self, target : TomatoElement | TomatoImage, screen : TomatoScreen | list[TomatoElement], cutoff : int=10, show_stats : bool=False) -> tuple[int, int]:

        start = time.time()

        target_matrix = target.get_image_matrix()
        target_width = target.get_width()
        target_height = target.get_height()

        distances_centers_elements = []

        if isinstance(screen, TomatoScreen):
            elements = screen.list_elements()
        elif isinstance(screen, list) and all(isinstance(e, TomatoElement) for e in screen):
            elements = screen
        else:
            raise TypeError("Screen must be a TomatoScreen or a list of TomatoElement instances.")


        for element in elements:

            element_matrix = element.get_image_matrix()
            element_center_relative_to_screen = element.get_center()
            element_center_relative_to_element = (element.get_width()//2, element.get_height()//2)

            element_section = self.__safe_section(element_matrix, 0, 0, target_height, target_width)
            if element_section is None: continue
            
            x_relative_to_element = (target_width // 2)
            y_relative_to_element = (target_height // 2)
            target_center_relative_to_element = (x_relative_to_element, y_relative_to_element)

            target_center_relative_to_screen = np.array(element_center_relative_to_screen) - (np.array(element_center_relative_to_element) - np.array(target_center_relative_to_element))
            target_center_relative_to_screen = tuple(int(axis) for axis in target_center_relative_to_screen)

            distance, center = (self.grayscale_image_distance(target_matrix, element_section), target_center_relative_to_screen)
            if distance<=cutoff and not show_stats:
                return center
            distances_centers_elements.append((distance, center, element))
            
        if len(distances_centers_elements)>0:
            closest_match = min(distances_centers_elements, key= lambda x: x[0])

            if show_stats:
                end = time.time()
                execution_time = f"{end - start:.2f}"
                print(f"Runtime: {execution_time}")
                print(f"Minimum Cutoff: {closest_match[0]}")
                closest_match[2].show()

            if closest_match[0] < cutoff: return closest_match[1]
        return(None, None)

        
            

