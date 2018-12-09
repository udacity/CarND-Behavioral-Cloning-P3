from keras.models import load_model
import h5py

file='model.h5'
model = load_model(file)

model.summary()

#from keras.utils import plot_model
#plot_model(model, to_file='model.png')


