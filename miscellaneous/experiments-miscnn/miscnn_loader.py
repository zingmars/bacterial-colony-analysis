# Experiment: Try using MIScnn
# The MIScnn library is a package, however, the wheel is incompatible with newer versions of Tensorflow
# To get it to work, we pull the original source in as a submodule and make python load it manually
import sys
import pathlib
print(f"Loading miscnn from `{pathlib.Path(__file__).parent.resolve()}\MIScnn\miscnn`!")
sys.path.append(f"{pathlib.Path(__file__).parent.resolve()}\MIScnn\miscnn")
import miscnn

# Create a Data I/O interface for kidney tumor CT scans in NIfTI format
from miscnn.data_loading.interfaces import NIFTI_interface
interface = NIFTI_interface(pattern="case_000[0-9]*", channels=1, classes=3)

# Initialize data path and create the Data I/O instance
data_path = ".\\data"
data_io = miscnn.Data_IO(interface, data_path)

sample_list = data_io.get_indiceslist()
sample_list.sort()

# Library import
from miscnn.processing.data_augmentation import Data_Augmentation

# Create and configure the Data Augmentation class
data_aug = Data_Augmentation(cycles=2, scaling=True, rotations=True, elastic_deform=True, mirror=True,
                             brightness=True, contrast=True, gamma=True, gaussian_noise=True)

from miscnn.processing.subfunctions.normalization import Normalization
from miscnn.processing.subfunctions.clipping import Clipping
from miscnn.processing.subfunctions.resampling import Resampling

# Create a pixel value normalization Subfunction through Z-Score 
sf_normalize = Normalization(mode='z-score')
# Create a clipping Subfunction between -79 and 304
sf_clipping = Clipping(min=-79, max=304)
# Create a resampling Subfunction to voxel spacing 3.22 x 1.62 x 1.62
sf_resample = Resampling((3.22, 1.62, 1.62))

# Assemble Subfunction classes into a list
# Be aware that the Subfunctions will be exectued according to the list order!
subfunctions = [sf_resample, sf_clipping, sf_normalize]

# Create a Preprocessor instance to configure how to preprocess the data into batches
pp = miscnn.Preprocessor(data_io, batch_size=4, analysis="patchwise-crop",
                         patch_shape=(128,128,128))
pp.patchwise_overlap = (40, 80, 80)

# Create a deep learning neural network model with a standard U-Net architecture
from miscnn.neural_network.architecture.unet.standard import Architecture
unet_standard = Architecture()
model = miscnn.Neural_Network(preprocessor=pp, architecture=unet_standard)

# Define Callbacks
from keras.callbacks import ReduceLROnPlateau
cb_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=20, verbose=1, mode='min', min_delta=0.0001, cooldown=1,    
                          min_lr=0.00001)
from keras.callbacks import EarlyStopping
cb_es = EarlyStopping(monitor='loss', min_delta=0, patience=150, verbose=1, mode='min')

# Create the validation sample ID list
validation_samples = sample_list

# Library import
from miscnn.evaluation.cross_validation import cross_validation
# Run cross-validation function
cross_validation(validation_samples, model, k_fold=3, epochs=1000, iterations=150,
                 evaluation_path="evaluation", draw_figures=True, callbacks=[cb_lr, cb_es])

