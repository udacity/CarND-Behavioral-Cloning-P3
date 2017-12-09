import unittest
import os
from train import *


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

if __name__ == '__main__':
    unittest.main()
