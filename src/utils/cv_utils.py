from __future__ import annotations
from abc import ABC, abstractmethod
from typing import AnyStr, List
import cv2
import numpy as np
import os

    
class ImageBrightnessHandler():
    
    def __init__(self, strategy: Strategy) -> None:
        self._strategy = strategy
    
    @property
    def strategy(self) -> Strategy:
        return self._strategy
    
    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        self._strategy = strategy
        
    def gamma_correction(self, image, gamma):
        look_up_table = np.empty((1,256), np.uint8)
        for i in range(256):
            look_up_table[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        return cv2.LUT(image, look_up_table)
    
    def adjust_images_brightness(self, images_path, gamma):
        images_list = self._strategy.select_images(images_path)
        for image in images_list :
            full_image_path = str(images_path + '/' + image)
            img_original = cv2.imread(full_image_path)
            img_adjusted = self.gamma_correction(img_original, gamma)
            cv2.imwrite(full_image_path, img_adjusted)


class Strategy(ABC):
    @abstractmethod
    def select_images(self, images_path: AnyStr) -> List:
        pass
    
class AllImageStrategy(Strategy):
    def select_images(self, images_path: AnyStr) -> List:
        return os.listdir(images_path)
    
class NoneImageStrategy(Strategy):
    def select_images(self, images_path: AnyStr) -> List:
        return []
    
class DarkImagesStrategy(Strategy):
    def __init__(self, percentile):
        self.percentile = percentile
    
    def compute_average_brightness(self, image):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value_channel = cv2.split(image_hsv)[2]
        hist = cv2.calcHist([value_channel], [0], None, [256], [0, 256])

        # Compute the average brightness
        total_brightness = np.sum(hist * np.arange(256))
        total_pixels = np.sum(hist)

        # Avoid division by zero
        average_brightness = total_brightness / total_pixels if total_pixels != 0 else 0

        return average_brightness
    
    def select_images(self, images_path: AnyStr) -> List:
        images_list = os.listdir(images_path)
        brightness_values = []

        # Calculate brightness for each image and collect values
        for image in images_list:
            full_image_path = str(images_path + '/' + image)
            img_original = cv2.imread(full_image_path)
            brightness = self.compute_average_brightness(img_original)
            brightness_values.append(brightness)

        # Calculate the first quartile
        first_quartile = np.percentile(brightness_values, self.percentile)

        # Select dark images
        selected_images = [image for image, brightness in zip(images_list, brightness_values) if brightness <= first_quartile]

        return selected_images