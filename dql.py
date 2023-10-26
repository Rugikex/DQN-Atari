from tensorflow import expand_dims
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D


class DeepQLearning(Model):
    def __init__(self, num_actions, activation='relu'):
        super(DeepQLearning, self).__init__()
        self.conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation=activation, input_shape=(84, 84, 4), padding='same')
        self.conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation=activation, input_shape=(84, 84, 4), padding='same')
        self.conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation=activation, input_shape=(84, 84, 4), padding='same')
        self.flatten = Flatten()
        self.fc1 = Dense(512, activation=activation)
        self.fc2 = Dense(num_actions)

    def call(self, inputs):
        # Add an extra dimension to the input to make it a batch of size 1
        x = expand_dims(inputs, axis=0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)[0] # Remove the extra dimension
