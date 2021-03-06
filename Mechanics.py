import MainNet
import Attention

# Load test image
test = '/path/to/testimage'


# Map prediction to weights file
def getweights():
    prediction = Attention.eval(test)

    if prediction == 'cell':
        return 'weights/cell_weights.h5'
    if prediction == 'brain':
        return 'weights/brain_tumor_weights.h5'
    if prediction == 'colon':
        return 'weights/colonoscopy_weights.h5'


# Apply weights to main network and execute
MainNet.execute(test, getweights())
