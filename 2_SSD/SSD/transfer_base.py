from ..VGG16.model import Model
from ssd512 import SSD512
from keras.layers.pooling.max_pooling2d import MaxPooling2D
from keras.layers.reshaping.zero_padding2d import ZeroPadding2D

# Load VGG
model = Model(data_folder="dataset_vgg")
model.build_regional_proposal_network(train_classes=14, train_vgg=True)
model.compile_model()
model.load_model(filename="vgg16.h5")

# Load SSD
# Parameter should be the same as final class count
ssd_model = SSD512(num_classes = 2)

or_model = model.get_model()
model_weights = or_model.get_weights()

ptr = 0
iii = 0
while (ptr < 26): # Only 13 layers should be transferred. The last 3 will be removed!
    if not isinstance(or_model.layers[iii], MaxPooling2D) and not isinstance(or_model.layers[iii], ZeroPadding2D): # They don't have weights!
        ssd_model.layers[iii+1].set_weights([model_weights[ptr], model_weights[ptr+1]]) # [ weights, bias ]
        ptr += 2
    iii += 1
   
ssd_model.save_weights("ssd.h5", overwrite=True)