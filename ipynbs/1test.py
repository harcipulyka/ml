import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import use as mpl_use

# get all valid backends
# mpl_use('throw an error but give me the available backends!')
mpl_use("MacOSX")

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def build_model(my_learning_rate):
    # Most simple tf.keras models are sequential.
    # A sequential model contains one or more layers.
    model = tf.keras.models.Sequential()

    # Describe the topography of the model.
    # The topography of a simple linear regression model
    # is a single node in a single layer.
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    # Compile the model topography into code that
    # TensorFlow can efficiently execute. Configure
    # training to minimize the model's mean squared error.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate), loss="mean_squared_error", metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(model, feature, label, epochs, batch_size):
    # Feed the feature values and the label values to the
    # model. The model will train for the specified number
    # of epochs, gradually learning how the feature values
    # relate to the label values.
    history = model.fit(x=feature, y=label, batch_size=batch_size, epochs=epochs)

    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the
    # rest of history.
    epochs = history.epoch

    # Gather the history (a snapshot) of each epoch.
    hist = pd.DataFrame(history.history)

    # Specifically gather the model's root mean
    # squared error at each epoch.
    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse


def plot_the_model(feature, label, prediction):
    # Plot the original data
    plt.scatter(feature, label, color='blue')

    # Plot the predicted values
    plt.plot(feature, prediction, color='red')

    # Set the title and axis labels
    plt.title('Label vs. Prediction')
    plt.xlabel('Feature')
    plt.ylabel('Label')

    plt.show()


def plot_the_loss_curve(epochs, rmse):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min()*0.97, rmse.max()])

    plt.show()


# sample data
my_feature = ([1.0, 2.0,  3.0,  4.0,  5.0,  6.0, 7.0,  8.0,  9.0, 10.0, 11.0, 12.0])
my_label = ([5.0, 8.8,  9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])

# first attempt
learning_rate = 0.3
epochs = 10
my_batch_size = 3

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature, my_label, epochs, my_batch_size)

prediction = my_model.predict(my_feature)

print("features: ", my_feature)
print("labels:", my_label)
print("my model: ", prediction)

plot_the_model(my_feature, my_label, prediction)
plot_the_loss_curve(epochs, rmse)
