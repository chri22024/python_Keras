from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

#データの読み込み
(train_images, train_classes), (_, _) = mnist.load_data()

#画像データ60000こ、画像サイズ28*28, 1チャンネル
train_images = train_images.reshape(60000, 28,28, 1)
#0〜1の範囲にスケーリングをする
train_images = train_images.astype('float32') / 255
# 正解ラベルには0〜9までのあるのでone-hot表現にする
train_classes = to_categorical(train_classes)

# 訓練・正解ラベルともに頭から1000個のデータを用いる
train_images = train_images[:1000]
train_classes = train_classes[:1000]

# Sequential モデルは、このような層を一直線に繋げて構成されるモデルであり、一つの入力から一つの出力を生成することができる
model = Sequential()

# 入力層（Input）
# 2次元データ対する畳み込みはConv2Dを使う

# (カーネル数、カーネルサイズ、データサイズが同じになるようにパディングを行うためにsame、活性化関数)
model.add(Conv2D(32, (3,3), padding = 'same', activation = 'relu', input_shape = (28, 28, 1)))
model.add(Conv2D(32, (3,3), padding = 'same', activation = 'relu' ))
model.add(MaxPooling2D(2,2))

# 畳み込み層（Conv1, Conv2）
model.add(Conv2D(64, (3,3), padding = 'same', activation = 'relu'))
model.add(Conv2D(64, (3,3), padding = 'same', activation = 'relu'))
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









