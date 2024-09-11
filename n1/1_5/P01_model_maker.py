from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import P10_util as util
import P11_model_util as mutil

class ModelMaker:
    

    def __init__(self, dst_dir, est_file, info_file, graph_file, input_size, hist_file,
                 filters, kernel_size, pool_size, dense_dims, lr, data_size, batch_size, epochs, vaild_rate):
        self.dst_dir = dst_dir
        self.est_file = est_file
        self.info_file = info_file
        self.graph_file =graph_file
        self.hist_file = hist_file
        self.input_size = input_size
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dense_dims = dense_dims
        self.lr = lr
        self.data_size = data_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.vaild_rata = vaild_rate


    def define_model(self):

        input_x = Input(shape=(*self.input_size, 1))
        x = input_x

        for f in self.filters:
            x = mutil.add_conv_pool_layers(x, f, self.kernel_size, self.pool_size)

        x = Flatten()(x)

        for dim in self.dense_dims[:-1]:
            x = mutil.add_dense_layer(x, dim)

        x = mutil.add_dense_layer(x, self.dense_dims[-1], activation='softmax')


        model = Model(input_x, x)


        model.compile(
            optimizer = Adam(learning_rate = self.lr),
            loss = 'categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    
    def fit_model(self):

        train_images, train_classes = util.load_data(self.data_size)

        model = self.define_model()


        history = model.fit(
            train_images,
            train_classes,
            batch_size = self.batch_size,
            epochs = self.epochs,
            validation_split = self.vaild_rata
        )

        return model, history.history
    

    def execute(self):

        model, history = self.fit_model()

        util.mkdir(self.dst_dir, rm=True)
        # model.save(self.est_file)
        model.save(self.est_file)
        util.plot(history, self.hist_file)

        mutil.save_model_info(self.info_file, self.graph_file, model)
        print('val_loss: %f' %history['val_loss'][-1])