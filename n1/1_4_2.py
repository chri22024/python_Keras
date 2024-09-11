from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Input # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

(train_images, train_classes), (_, _) = mnist.load_data()

train_images = train_images.reshape(60000, 28,28, 1)
train_images = train_images.astype('float32') / 255
train_classes = to_categorical(train_classes)

train_images = train_images[:1000]
train_classes = train_classes[:1000]

input_x = Input(shape=(28,28,1))
x = input_x

x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)





x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)


x = Dense(1024, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(10, activation='softmax')(x)


model = Model(input_x, x)



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









