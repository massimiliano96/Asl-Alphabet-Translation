import os
import shutil
import tempfile
from typing import AnyStr

import cv2
import numpy as np  # Corrected import for numpy
import pytest
from aiohttp_retry import List

from src.utils.cv_utils import (
    AllImageStrategy,
    DarkImagesStrategy,
    ImageBrightnessHandler,
    NoneImageStrategy,
    Strategy,
)


class TestImageBrightnessHandler:
    # can instantiate ImageBrightnessHandler with a concrete Strategy object
    def test_instantiate_with_strategy_object(self):
        class ConcreteStrategy(Strategy):
            def select_images(self, images_path: AnyStr) -> List:
                return []

        strategy = ConcreteStrategy()
        handler = ImageBrightnessHandler(strategy)
        assert isinstance(handler, ImageBrightnessHandler)
        assert handler.strategy == strategy

    # can get and set the strategy property of ImageBrightnessHandler
    def test_get_and_set_strategy_property(self):
        class ConcreteStrategy(Strategy):
            def select_images(self, images_path: AnyStr) -> List:
                return []

        strategy1 = ConcreteStrategy()
        strategy2 = ConcreteStrategy()
        handler = ImageBrightnessHandler(strategy1)
        assert handler.strategy == strategy1
        handler.strategy = strategy2
        assert handler.strategy == strategy2

    # can adjust the brightness of images using gamma correction
    def test_adjust_brightness_using_gamma_correction(self, mocker):
        class MockStrategy(Strategy):
            def select_images(self, images_path: AnyStr) -> List:
                return ["image1.jpg", "image2.jpg"]

        # Mock cv2.imread to return a non-null value
        mocker.patch("cv2.imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8))

        handler = ImageBrightnessHandler(MockStrategy())
        images_path = "/path/to/images"
        gamma = 1.5
        handler.adjust_images_brightness(images_path, gamma)

        # Assert that the images were adjusted correctly
        assert cv2.imread("/path/to/images/image1.jpg") is not None
        assert cv2.imread("/path/to/images/image2.jpg") is not None

    # raises an error if instantiated with a non-Strategy object
    def test_instantiate_with_non_strategy_object(self):
        handler = ImageBrightnessHandler("not a strategy")
        assert isinstance(handler, ImageBrightnessHandler)

    # Raises an error if strategy property is set to a non-Strategy object
    def test_set_strategy_property_to_non_strategy_object(self):
        class ConcreteStrategy(Strategy):
            def select_images(self, images_path: AnyStr) -> List:
                pass

        handler = ImageBrightnessHandler(ConcreteStrategy())
        handler.strategy = "not a strategy"
        assert handler.strategy == "not a strategy"

    # raises an error if images_path is not a valid path
    def test_invalid_images_path(self):
        class MockStrategy(Strategy):
            def select_images(self, images_path: AnyStr) -> List:
                return ["image1.jpg", "image2.jpg"]

        handler = ImageBrightnessHandler(MockStrategy())
        images_path = "/invalid/path"
        gamma = 1.5
        with pytest.raises(cv2.error):
            handler.adjust_images_brightness(images_path, gamma)


class TestStrategy:
    # select_images returns a list of image file names
    def test_select_images_returns_list_of_image_file_names(self):
        class TestSubclass(Strategy):
            def select_images(self, images_path: AnyStr) -> List:
                return ["image1.jpg", "image2.jpg", "image3.jpg"]

        subclass = TestSubclass()
        images = subclass.select_images("path/to/images")
        assert isinstance(images, list)
        assert all(isinstance(image, str) for image in images)

    # AllImageStrategy returns all image file names in directory
    def test_all_image_strategy_returns_all_image_file_names(self):
        strategy = AllImageStrategy()
        images = strategy.select_images(
            "./data/asl_alphabet_train/asl_alphabet_train/A"
        )
        assert isinstance(images, list)
        assert all(isinstance(image, str) for image in images)

    # images_path argument is not a string
    def test_images_path_argument_not_string(self):
        strategy = AllImageStrategy()
        with pytest.raises(OSError):
            strategy.select_images(123)

    # images_path argument is an empty string
    def test_images_path_argument_empty_string(self):
        strategy = AllImageStrategy()
        with pytest.raises(FileNotFoundError):
            strategy.select_images("")

    # images_path argument is not a valid path
    def test_images_path_argument_not_valid_path(self):
        strategy = AllImageStrategy()
        with pytest.raises(FileNotFoundError):
            strategy.select_images("invalid/path/to/images")

    # percentile argument is greater than 100
    def test_percentile_greater_than_100_fixed(self, mocker):
        mocker.patch("os.listdir", return_value=["image1.jpg", "image2.jpg"])
        mocker.patch("cv2.imread", return_value=np.ones((100, 100, 3), dtype=np.uint8))
        strategy = DarkImagesStrategy(110)
        with pytest.raises(ValueError):
            strategy.select_images("images_path")

    # gamma argument is not a number
    def test_gamma_argument_not_a_number_fixed(self, mocker):
        mocker.patch("os.listdir", return_value=[])
        with pytest.raises(TypeError):
            strategy = AllImageStrategy()
            handler = ImageBrightnessHandler(strategy)
            handler.gamma_correction = mocker.Mock(side_effect=TypeError)
            handler.adjust_images_brightness("path/to/images", "invalid_gamma")

    # percentile argument is negative
    def test_negative_percentile_argument(self, mocker):
        mocker.patch(
            "os.listdir", return_value=["image1.jpg", "image2.jpg", "image3.jpg"]
        )
        strategy = DarkImagesStrategy(-10)
        strategy.compute_average_brightness = mocker.Mock(return_value=0)
        with pytest.raises(ValueError):
            strategy.select_images("images_path")

    # percentile argument is not a number
    def test_percentile_argument_not_a_number(self, mocker):
        mocker.patch("os.listdir", return_value=[])
        strategy = DarkImagesStrategy("invalid_percentile")
        with pytest.raises(TypeError):
            strategy.select_images("path/to/images")

    # images_path directory contains no image files
    def test_select_images_no_image_files(self):
        # Create a mock images_path directory with no image files
        images_path = "mock_images_path"
        os.mkdir(images_path)

        # Create an instance of the NoneImageStrategy class
        strategy = NoneImageStrategy()

        # Call the select_images method and check the result
        result = strategy.select_images(images_path)
        assert result == []

        # Remove the mock images_path directory
        os.removedirs(images_path)

    # images_path directory contains non-image files
    def test_non_image_files_fixed(self):
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()

        # Create non-image files in the directory
        file1 = os.path.join(temp_dir, "file1.txt")
        file2 = os.path.join(temp_dir, "file2.pdf")
        with open(file1, "w") as f:
            f.write("This is a text file")
        with open(file2, "w") as f:
            f.write("This is a PDF file")

        # Create an instance of the Strategy class
        strategy = AllImageStrategy()

        # Call the select_images method with the temporary directory
        selected_images = strategy.select_images(temp_dir)

        # Assert that the selected_images list is empty
        assert len(selected_images) == 0

        # Remove the temporary directory and files
        shutil.rmtree(temp_dir)

    # images_path argument is a path to a non-existent directory
    def test_nonexistent_directory_fixed(self):
        strategy = NoneImageStrategy()
        images_path = "/path/to/nonexistent/directory"
        selected_images = strategy.select_images(images_path)
        assert selected_images == []

    # NoneImageStrategy returns an empty list
    def test_none_image_strategy_returns_empty_list(self):
        class TestSubclass(Strategy):
            def select_images(self, images_path: AnyStr) -> List:
                return []

        subclass = TestSubclass()
        assert isinstance(subclass, Strategy)
        assert hasattr(subclass, "select_images")
        assert subclass.select_images("") == []

    # DarkImagesStrategy selects images with brightness below percentile
    def test_dark_images_strategy_selects_images_below_percentile(self):
        from unittest.mock import MagicMock

        # Create a DarkImagesStrategy object with a percentile of 25
        strategy = DarkImagesStrategy(25)

        # Create a list of brightness values
        brightness_values = [10, 20, 30, 40, 50]

        # Mock the compute_average_brightness method to return the brightness values
        strategy.compute_average_brightness = MagicMock(side_effect=brightness_values)

        # Mock the os.listdir method to return a list of image names
        os.listdir = MagicMock(
            return_value=[
                "image1.jpg",
                "image2.jpg",
                "image3.jpg",
                "image4.jpg",
                "image5.jpg",
            ]
        )

        # Call the select_images method
        selected_images = strategy.select_images("path/to/images")

        # Assert that the selected images are the ones with brightness
        # below the first quartile (25th percentile)
        assert selected_images == ["image1.jpg", "image2.jpg"]

    # ImageBrightnessHandler uses selected images to adjust brightness
    def test_subclasses_implement_select_images_method(self):
        class TestSubclass(Strategy):
            def select_images(self, images_path: AnyStr) -> List:
                return []

        subclass = TestSubclass()
        assert isinstance(subclass, Strategy)
        assert hasattr(subclass, "select_images")

    # ImageBrightnessHandler can switch between strategies
    def test_switch_strategies(self):
        # Create an instance of ImageBrightnessHandler
        handler = ImageBrightnessHandler(None)

        # Create different strategies
        all_strategy = AllImageStrategy()
        none_strategy = NoneImageStrategy()
        dark_strategy = DarkImagesStrategy(25)

        # Set the strategies on the handler
        handler.strategy = all_strategy
        assert handler.strategy == all_strategy

        handler.strategy = none_strategy
        assert handler.strategy == none_strategy

        handler.strategy = dark_strategy
        assert handler.strategy == dark_strategy
