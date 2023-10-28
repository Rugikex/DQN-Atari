from tensorflow import expand_dims
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D


class DeepQNetwork(Model):
    def __init__(self, num_actions, activation='relu') -> None:
        super(DeepQNetwork, self).__init__()
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

    def get_config(self):
        config = {
            'num_actions': self.fc2.units,
            'activation': self.conv1.activation.__name__
        }
        base_config = super(DeepQNetwork, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
