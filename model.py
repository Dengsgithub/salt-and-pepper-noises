from keras.models import Model
from keras.layers import Input, Add, PReLU, Conv2DTranspose, Concatenate, MaxPooling2D, UpSampling2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf
from keras.layers.merge import concatenate
from keras.layers.core import Lambda
from keras.layers import merge, Activation

class L0Loss:
    def __init__(self):
        self.gamma = K.variable(2.)

    def __call__(self):
        def calc_loss(y_true, y_pred):
            loss = K.pow(K.abs(y_true - y_pred) + 1e-8, self.gamma)
            return loss
        return calc_loss


class UpdateAnnealingParameter(Callback):
    def __init__(self, gamma, nb_epochs, verbose=0):
        super(UpdateAnnealingParameter, self).__init__()
        self.gamma = gamma
        self.nb_epochs = nb_epochs
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        new_gamma = 2.0 * (self.nb_epochs - epoch) / self.nb_epochs
        K.set_value(self.gamma, new_gamma)

        if self.verbose > 0:
            print('\nEpoch %05d: UpdateAnnealingParameter reducing gamma to %s.' % (epoch + 1, new_gamma))


def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNR(y_true, y_pred):
    max_pixel = 255.0
    y_pred = K.clip(y_pred, 0.0, 255.0)
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))


def get_model(model_name="srresnet"):
    if model_name == "srresnet":
        return get_srresnet_model()
    elif model_name == "unet":
        return get_unet_model(out_ch=3)
    elif model_name == "xin":
        return get_lian_model()
    else:
        raise ValueError("model_name should be 'srresnet'or 'unet'")


def get_lian_model(pretrained_weights=None, input_size=(None, None, 3)):
    n_init_features = 64

    def min_pool2d(x):
        min_x = -K.pool2d(-x, pool_size=(2, 2), strides=(1, 1), padding='same')
        return min_x

    def find_medians(x, k=3):
        patches = tf.extract_image_patches(
            x,
            ksizes=[1, k, k, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='SAME')
        m_idx = int(k * k / 2 + 1)
        top, _ = tf.nn.top_k(patches, m_idx, sorted=True)
        median = tf.slice(top, [0, 0, 0, m_idx - 1], [-1, -1, -1, 1])
        return median

    def median_pool2d(x, k=3):
        channels = tf.split(x, num_or_size_splits=x.shape[3], axis=3)
        for channel in channels:
            channel = find_medians(channel, k)
        #median = merge(channels, mode='concat', concat_axis=-1)
        median = concatenate(channels, axis=-1)
        return median

    def min_pool2d_output_shape(input_shape):
        shape = list(input_shape)
        return tuple(shape)

    def _residual_block(inputs, feature_dim=64):
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        m = Add()([x, inputs])
        return m

    input_img = Input(shape=input_size, name='input_image')
    # x5 = Lambda(median_pool2d, arguments={'k': 3},
    #             output_shape=min_pool2d_output_shape)(input_img)
    # x = x5
    x = Conv2D(n_init_features, (3, 3), padding="same", kernel_initializer="he_normal")(input_img)
    x = PReLU(shared_axes=[1, 2])(x)

    for i in range(16):
        x = Conv2D(n_init_features, (3, 3), kernel_initializer='Orthogonal', padding='same')(x)
        x = BatchNormalization(axis=3, momentum=0.99, epsilon=0.0001)(x)
        x = Activation('relu')(x)
        x = _residual_block(x, feature_dim=n_init_features)
    for j in range(16):
        x = Conv2D(n_init_features, (5, 5), kernel_initializer='Orthogonal', padding='same')(x)
        x = BatchNormalization(axis=3, momentum=0.99, epsilon=0.0001)(x)
        x = Activation('relu')(x)
        x = _residual_block(x, feature_dim=n_init_features)

    '''
    for i in range(8):
        x = Conv2DTranspose(n_init_features, (3, 3), kernel_initializer='Orthogonal', padding='same')(x)
        x = BatchNormalization(axis=3, momentum=0.99, epsilon=0.0001)(x)
        x = Activation('relu')(x) 
    '''
    x = Conv2D(3, (3, 3), kernel_initializer='Orthogonal', padding='same')(x)
    model = Model(input=input_img, output=x)
    # model = Model(input=[input_img, input_loc], output=x)
    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

# SRResNet
def get_srresnet_model(input_channel_num=3, feature_dim=64, resunit_num=16):
    def _residual_block(inputs):
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        m = Add()([x, inputs])

        return m

    inputs = Input(shape=(None, None, input_channel_num))
    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    x0 = x

    for i in range(resunit_num):
        x = _residual_block(x)

    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Add()([x, x0])
    x = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    model = Model(inputs=inputs, outputs=x)

    return model


# UNet: code from https://github.com/pietz/unet-keras
def get_unet_model(input_channel_num=3, out_ch=3, start_ch=64, depth=4, inc_rate=2., activation='relu',
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
    def _conv_block(m, dim, acti, bn, res, do=0):
        n = Conv2D(dim, 3, activation=acti, padding='same')(m)
        n = BatchNormalization()(n) if bn else n
        n = Dropout(do)(n) if do else n
        n = Conv2D(dim, 3, activation=acti, padding='same')(n)
        n = BatchNormalization()(n) if bn else n

        return Concatenate()([m, n]) if res else n

    def _level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
        if depth > 0:
            n = _conv_block(m, dim, acti, bn, res)
            m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
            m = _level_block(m, int(inc * dim), depth - 1, inc, acti, do, bn, mp, up, res)
            if up:
                m = UpSampling2D()(m)
                m = Conv2D(dim, 2, activation=acti, padding='same')(m)
            else:
                m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
            n = Concatenate()([n, m])
            m = _conv_block(n, dim, acti, bn, res)
        else:
            m = _conv_block(m, dim, acti, bn, res, do)

        return m

    i = Input(shape=(None, None, input_channel_num))
    o = _level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv2D(out_ch, 1)(o)
    model = Model(inputs=i, outputs=o)

    return model


def main():
    # model = get_model()
    model = get_model("unet")
    model.summary()


if __name__ == '__main__':
    main()
