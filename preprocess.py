from math import ceil
import sys

from model import *
from pickle import dump
from operator import itemgetter

batch_size = 2 ** 10
columns = load_driving_log('data/driving_log.csv')

validation_ratio = 0.2
total_records = columns[0].shape[0]

columns = shuffle(*columns)

validation_set_count = int(ceil(total_records * validation_ratio))
validation_set = list(map(itemgetter(slice(0, validation_set_count)), columns))
training_set_count = total_records - validation_set_count
training_set = list(map(itemgetter(slice(validation_set_count, total_records)), columns))

print('Total', total_records)
print('Validation', validation_set_count)
print('Training', training_set_count)

if not os.path.isdir('data_cache'):
    os.mkdir('data_cache')

augmented_training_set_count = 0
for i in range(training_set_count // batch_size + 1):
    end = min((i + 1) * batch_size, training_set_count)
    left, center, right, steering = map(itemgetter(slice(i * batch_size, end)),
                                        training_set)
    images, steering = make_training_batch(left, center, right, steering)
    with open('data_cache/training-{}.p'.format(i), 'wb') as file:
        dump((images, steering), file)

    sys.stdout.write('Processed {} of {}\n'.format(end, training_set_count))
    sys.stdout.flush()
    augmented_training_set_count += steering.shape[0]

for i in range(validation_set_count // batch_size + 1):
    end = min((i + 1) * batch_size, validation_set_count)
    left, center, right, steering = map(
        itemgetter(slice(i * batch_size, end)),
        training_set)
    images = crop_image(preprocess_images(load_images(center)))
    with open('data_cache/validation-{}.p'.format(i), 'wb') as file:
        dump((images, steering), file)

    sys.stdout.write('Processed {} of {}\n'.format(end, validation_set_count))
    sys.stdout.flush()

print('Wrote {} training points.'.format(augmented_training_set_count))
