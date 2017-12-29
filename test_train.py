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


if __name__ == '__main__':
    unittest.main()
