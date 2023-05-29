# Exports a picture of how the model is set up.
# The pic is kind of big tho
from model import Model
from keras.utils import plot_model

model = Model()
model.build_regional_proposal_network(train_classes=14, train_vgg=True)
model.compile_model()
model.load_model(filename="vgg16.h5")
plot_model(model.get_model(), "ssd.jpeg", show_shapes = True)