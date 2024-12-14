from tensorflow.keras.layers import (Input, Conv2D, MaxPool2D, BatchNormalization, Lambda, Dense, 
                                     Bidirectional, LSTM)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

class OCRModel:
    """Class to contain the functionality of the OCR model

    Attributes:
        model (Model): The OCR model
    
    Methods:
        build_model(): Build the OCR model
        load_weights(weights_path): Load the weights of the model
        predict(image): Make a prediction on the provided image
    """

    def __init__(self, weights_path="./api/model/model_weights/working_old_weights.hdf5"):
        # Build the model
        self.model = self.build_model()

        # Load the weights from the saved weights
        if weights_path:
            self.load_weights(weights_path)

    def build_model(self):
        # Input with shape of height=40 and width=140
        inputs = Input(shape=(40, 140, 1))

        # Convolutional and pooling layers
        conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

        conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
        pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

        conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)

        conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)
        pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

        conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
        batch_norm_5 = BatchNormalization()(conv_5)

        conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
        batch_norm_6 = BatchNormalization()(conv_6)
        pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

        conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)

        squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

        # Bidirectional LSTM layers
        blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(squeezed)
        blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(blstm_1)

        # Output layer
        outputs = Dense(36, activation='softmax')(blstm_2)

        # Create the model
        model = Model(inputs, outputs)

        return model

    def load_weights(self, weights_path):
        # Load the weights into the model
        self.model.load_weights(weights_path)
        print(f"Weights loaded from {weights_path}")

    def predict(self, image):
        # Use the model to make a prediction on the provided image
        return self.model.predict(image)