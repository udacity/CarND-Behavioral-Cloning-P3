import unittest
import os
from train import *
import cv2


class SDCSimulationTrain(unittest.TestCase):

    def test_tensorflow_tags(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.assertEqual(os.environ['TF_CPP_MIN_LOG_LEVEL'], '2')

    def test_load_configuration(self):
        conf_path = 'test/test_configuration.json'
        tester_configuration = { "use_grid_search": "False",
                                 "input_path": "test/test_data",
                                 "output_path": "test/test_model",
                                 "loss_function": "mse",
                                 "epochs": 75,
                                 "dropout": 0.5,
                                 "use_tensorboard": "True",
                                 "units": 1,
                                 "tensorboard_log_dir": "test/test_logs",
                                 "old_image_root": "test/test_data",
                                 "new_image_root": "test/test_data"
                                 }
        loaded_configuration = load_config(conf_path)
        self.assertDictEqual(loaded_configuration, tester_configuration)

    def test_get_file_list(self):
        directory_path = 'test/test_data'
        test_file_list = ['test/test_data/inside_in_grass_fast/driving_log.csv',
                      'test/test_data/inside_just_at_curb_good/driving_log.csv']
        file_list = get_file_list(directory_path)
        self.assertEqual(file_list, test_file_list)

    def test_get_log_lines(self):
        test_file_list = ['test/test_data/inside_in_grass_fast/driving_log.csv',
                      'test/test_data/inside_just_at_curb_good/driving_log.csv']
        test_lines = []
        [test_lines.append([path, get_log_lines(path)]) for path in test_file_list]
        reference_value = '30.19097'
        self.assertEqual(test_lines[1][1][1][6], reference_value)

    def test_get_measurements_and_measurements(self):
        pass

    def test_augment_brightness_camera_images(self):
        image_path = 'test/test_data/inside_in_grass_fast/IMG/center_2017_11_17_10_08_33_895.jpg'
        image = cv2.imread(image_path)
        brightened_image = augment_brightness_camera_images(image)
        # Assert that the same shape was returned. Can do more intensive checks later.
        self.assertEqual(image.shape[0], brightened_image.shape[0])
        self.assertEqual(image.shape[1], brightened_image.shape[1])
        self.assertEqual(image.shape[2], brightened_image.shape[2])

    def test_use_sides_for_recovery(self):
        reference_value = 30.19097
        adjustment = .25
        left_value = reference_value + adjustment
        left_test = adjust_side_images(reference_value, adjustment, 'left')
        self.assertEqual(left_value, left_test)
        right_value = reference_value - adjustment
        right_test = adjust_side_images(reference_value, adjustment, 'right')
        self.assertEqual(right_value, right_test)
        self.assertEqual(reference_value, adjust_side_images(reference_value, adjustment, 'center'))

    def test_shift_image_position(self):
        image_path = 'test/test_data/inside_in_grass_fast/IMG/center_2017_11_17_10_08_33_895.jpg'
        image = cv2.imread(image_path)
        steering_angle = 30.67882
        test_translation_range = 0.004
        test_image, test_angle = shift_image_position(image, steering_angle, test_translation_range)
        self.assertNotAlmostEqual(steering_angle, test_angle)
        # Assert images of the same shape were returned
        self.assertEqual(image.shape[0], test_image.shape[0])
        self.assertEqual(image.shape[1], test_image.shape[1])
        self.assertEqual(image.shape[2], test_image.shape[2])

    def test_add_random_shadow(self):
        image_path = 'test/test_data/inside_in_grass_fast/IMG/center_2017_11_17_10_08_33_895.jpg'
        image = cv2.imread(image_path)
        shadow_image = add_random_shadow(image)
        # Assert images of the same shape were returned
        self.assertEqual(image.shape[0], shadow_image.shape[0])
        self.assertEqual(image.shape[1], shadow_image.shape[1])
        self.assertEqual(image.shape[2], shadow_image.shape[2])

    def test_flip_image(self):
        image_path = 'test/test_data/inside_in_grass_fast/IMG/center_2017_11_17_10_08_33_895.jpg'
        image = cv2.imread(image_path)
        flipped_image = flip_image(image)
        # Assert images of the same shape were returned
        self.assertEqual(image.shape[0], flipped_image.shape[0])
        self.assertEqual(image.shape[1], flipped_image.shape[1])
        self.assertEqual(image.shape[2], flipped_image.shape[2])

    def test_crop_image(self):
        image_path = 'test/test_data/inside_in_grass_fast/IMG/center_2017_11_17_10_08_33_895.jpg'
        image = cv2.imread(image_path)
        horizon_divisor = 5
        hood_pixels = 25
        crop_height = 64
        crop_width = 64
        cropped_image = crop_image(image, horizon_divisor, hood_pixels, crop_height, crop_width)
        # Assert returned image's shape
        self.assertEqual(cropped_image.shape[0], crop_height)
        self.assertEqual(cropped_image.shape[1], crop_width)
        self.assertEqual(cropped_image.shape[2], 3)


if __name__ == '__main__':
    unittest.main()
