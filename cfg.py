# configure the environment
use_gpu = False or True  # put a # after False to disable GPU usage
debug_show_example_images = 0  # 0: off, >0: number of show events

# define input data
first_dataset = 1  # defines the first subdirectory, see formatting below
last_dataset = 4  # defines the last subdirectory (inclusive)
data_root_path_fmt = './data/{:02}/'
csv_rel_path = 'driving_log.csv'
img_rel_path = 'IMG/'

# input data manipulation
generator_new_item_multiplier = 2  # generator will output twice the amount of its input by adding flipped images
camera_steer_offset = 0.2  # this camera offset will be applied to left and right camera images (multiplied by -1. for the latter)
camera_steer_multiplier = 1.0  # multiplies the steering angle
ignore_zero_steer_items_from = 10 # from this number of items... (see next line)
ignore_zero_steer_items_skip = 8   # ...skip this number of items (prev line contd)
enable_datagen = False # or True
datagen_item_multiplier = 2  # doubles the original input records
datagen_rotation_range = 5
test_size = 0.2

# set up the model
batch_size = 32
epochs = 8
model_rel_path = 'model.h5'

class state:
    debug_show_example_images_count = 0
    ignore_zero_steer_items_count = 0

