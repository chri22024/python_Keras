from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

(train_images, train_classes), (_, _) = mnist.load_data()

train_images = train_images.reshape(60000, 28,28, 1)
train_images = train_images.astype('float32') / 255
train_classes = to_categorical(train_classes)

train_images = train_images[:1000]
train_classes = train_classes[:1000]


model = Sequential()


model.add(Conv2D(32, (3,3), padding = 'same', activation = 'relu', input_shape = (28, 28, 1)))
model.add(Conv2D(32, (3,3), padding = 'same', activation = 'relu', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(2,2))


model.add(Conv2D(64, (3,3), padding = 'same', activation = 'relu', input_shape = (28, 28, 1)))
model.add(Conv2D(64, (3,3), padding = 'same', activation = 'relu', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(2,2))


model.add(Flatten())

model.add(Dense(1024, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))


model.compile(
    optimizer = Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics = ['accuracy']
)



history = model.fit(
    train_images,
    train_classes,
    batch_size = 128,
    epochs=10,
    validation_split=0.2
)

print('val_loss: %f' % history.history['val_loss'][-1])









