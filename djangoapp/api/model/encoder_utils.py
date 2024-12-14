import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import backend as K

def load_encoder():
    """Load the LabelEncoder from the saved file

    Returns:
        LabelEncoder: The loaded LabelEncoder
    """
    char_encoder = LabelEncoder()
    char_encoder.classes_ = np.load('./api/model/label_encoder/classes.npy')
    return char_encoder

def decode_characters(prediction):
    """Decode the given encoded labels

    Args:
        prediction (numpy.ndarray): The encoded labels

    Returns:
        str: The decoded characters
    """
    char_encoder = load_encoder()
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],greedy=True)[0][0])
    for x in out:
        x  = [y for y in x if y != -1]
        decoded_sequence = char_encoder.inverse_transform(x)
        result = ''.join(map(str, decoded_sequence))

    return result