# configure the environment
GPU = False or True  # put a # after False to disable GPU usage

# define input data
first_dataset = 1  # defines the first subdirectory, see formatting below
last_dataset = 4  # defines the last subdirectory (inclusive)
data_root_path_fmt = './data/{:02}/'
csv_rel_path = 'driving_log.csv'
img_rel_path = 'IMG/'

# input data manipulation
generator_new_item_multiplier = 2  # generator will output twice the amount of its input by adding flipped images
camera_steer_offset = 0.3  # this camera offset will be applied to left and right camera images (multiplied by -1. for the latter)
test_size = 0.2

# set up the model
batch_size = 32
epochs = 2
model_rel_path = 'model.h5'
