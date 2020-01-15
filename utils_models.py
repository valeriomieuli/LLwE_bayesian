import os
import tensorflow.keras as keras
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend


def build_features_extractor(model_name, input_shape):
    if model_name == 'VGG16':
        model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'VGG19':
        model = applications.VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'ResNet50':
        model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        assert (False), "Specified base model is not available !"

    x = model.output
    x = keras.layers.Flatten()(x)  # keras.layers.GlobalMaxPooling2D()(x)
    model = keras.Model(inputs=model.input, outputs=x)

    return model


def build_autoencoder(n_input_features, hidden_layer_size, weight_decay):
    input = keras.layers.Input(shape=(n_input_features,))
    x = keras.layers.Dense(units=hidden_layer_size, activation=None,
                           kernel_regularizer=keras.regularizers.l2(weight_decay))(input)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Dense(units=n_input_features, activation=None,
                           kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = keras.layers.Activation(activation='sigmoid')(x)
    model = keras.Model(inputs=input, outputs=x)

    return model


def get_preprocessing_function(model_name):
    if model_name == 'InceptionResNetV2':
        return applications.inception_resnet_v2.preprocess_input
    elif model_name == 'VGG16':
        return applications.vgg16.preprocess_input
    elif model_name == 'VGG19':
        return applications.vgg19.preprocess_input
    elif model_name in ['ResNet50', 'ResNet18']:
        return applications.resnet50.preprocess_input
    else:
        raise ValueError("Preprocessing function for the specified base model is not available !")


def build_expert(model_name, input_shape, n_classes, weight_decay):
    if model_name == 'InceptionResNetV2':
        base_model = applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',
                                                                        input_shape=input_shape)
        head_model = keras.layers.GlobalAveragePooling2D()(base_model.output)
        head_model = keras.layers.Dense(units=n_classes, activation="softmax")(head_model)
    elif model_name == 'VGG16':
        base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

        head_model = keras.layers.Flatten()(base_model.output)
        head_model = keras.layers.Dense(units=1024, activation='relu')(head_model)
        head_model = keras.layers.Dense(units=1024, activation='relu')(head_model)
        head_model = keras.layers.Dense(units=n_classes, activation='softmax')(head_model)
    elif model_name == 'VGG19':
        base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

        head_model = keras.layers.Flatten()(base_model.output)
        head_model = keras.layers.Dense(units=1024)(head_model)
        head_model = keras.layers.Dense(units=1024)(head_model)
        head_model = keras.layers.Dense(units=n_classes, activation='softmax')(head_model)
    else:
        raise ValueError("Specified base model is not available !")

    model = keras.Model(inputs=base_model.input, outputs=head_model)
    if weight_decay != -1:
        for layer in model.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = keras.regularizers.l2(weight_decay)

    return model


def build_bayesian_model(model_name, input_shape, n_classes, weight_decay, dropout_rate, seed):
    if model_name in ["VGG16", "VGG19"]:
        model = build_expert(model_name=model_name, input_shape=input_shape, n_classes=n_classes,
                             weight_decay=weight_decay)
        x = model.layers[0].output
        for i in range(1, len(model.layers)):
            if isinstance(model.layers[i], keras.layers.Conv2D) or isinstance(model.layers[i], keras.layers.Dense):
                x = keras.layers.Dropout(rate=dropout_rate, seed=seed)(x, training=True)
            x = model.layers[i](x)
        bayesian_model = keras.Model(inputs=model.layers[0].input, outputs=x)

    elif model_name == "InceptionResNetV2":
        bayesian_model = bayesian_InceptionResNetV2(weights='imagenet', input_shape=input_shape, classes=n_classes,
                                                    dropout_rate=dropout_rate, seed=seed)

    elif model_name == "ResNet50":
        bayesian_model = bayesian_ResNet50(weights='imagenet', input_shape=input_shape, classes=n_classes,
                                           dropout_rate=dropout_rate, seed=seed)

    elif model_name == "ResNet18":
        bayesian_model = bayesian_ResNet18(input_shape=input_shape, dropout_rate=dropout_rate, seed=seed,
                                           classes=n_classes)
    else:
        raise ValueError("Specified base model is not available !")

    # Apply L2-regularization if required
    if weight_decay != -1:
        for layer in bayesian_model.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = keras.regularizers.l2(weight_decay)

    return bayesian_model


'''def bayesian_InceptionResNetV2(weights='imagenet', input_shape=(256, 256, 3),
                               classes=1000, dropout_rate=0.5, seed=1234):
    
     #code based on the one from:
     #https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_resnet_v2.py
    

    def conv2d_bn(x, filters, kernel_size, strides=1, padding='same', activation='relu',
                  use_bias=False, name=None, dropout_rate=0.5, seed=1234):
        if dropout_rate != -1:
            x = layers.Dropout(rate=dropout_rate, seed=seed)(x, training=True)
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, name=name)(x)
        if not use_bias:
            bn_axis = 1 if backend.image_data_format() == 'channels_first' else 3
            bn_name = None if name is None else name + '_bn'
            x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
            if activation is not None:
                ac_name = None if name is None else name + '_ac'
            x = layers.Activation(activation, name=ac_name)(x)
        return x

    def inception_resnet_block(x, scale, block_type, block_idx, activation='relu', dropout_rate=0.5, seed=1234):
        if block_type == 'block35':
            branch_0 = conv2d_bn(x, 32, 1, dropout_rate=dropout_rate, seed=seed)
            branch_1 = conv2d_bn(x, 32, 1, dropout_rate=dropout_rate, seed=seed)
            branch_1 = conv2d_bn(branch_1, 32, 3, dropout_rate=dropout_rate, seed=seed)
            branch_2 = conv2d_bn(x, 32, 1, dropout_rate=dropout_rate, seed=seed)
            branch_2 = conv2d_bn(branch_2, 48, 3, dropout_rate=dropout_rate, seed=seed)
            branch_2 = conv2d_bn(branch_2, 64, 3, dropout_rate=dropout_rate, seed=seed)
            branches = [branch_0, branch_1, branch_2]
        elif block_type == 'block17':
            branch_0 = conv2d_bn(x, 192, 1, dropout_rate=dropout_rate, seed=seed)
            branch_1 = conv2d_bn(x, 128, 1, dropout_rate=dropout_rate, seed=seed)
            branch_1 = conv2d_bn(branch_1, 160, [1, 7], dropout_rate=dropout_rate, seed=seed)
            branch_1 = conv2d_bn(branch_1, 192, [7, 1], dropout_rate=dropout_rate, seed=seed)
            branches = [branch_0, branch_1]
        elif block_type == 'block8':
            branch_0 = conv2d_bn(x, 192, 1, dropout_rate=dropout_rate, seed=seed)
            branch_1 = conv2d_bn(x, 192, 1, dropout_rate=dropout_rate, seed=seed)
            branch_1 = conv2d_bn(branch_1, 224, [1, 3], dropout_rate=dropout_rate, seed=seed)
            branch_1 = conv2d_bn(branch_1, 256, [3, 1], dropout_rate=dropout_rate, seed=seed)
            branches = [branch_0, branch_1]
        else:
            raise ValueError('Unknown Inception-ResNet block type. '
                             'Expects "block35", "block17" or "block8", '
                             'but got: ' + str(block_type))

        block_name = block_type + '_' + str(block_idx)
        channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
        mixed = layers.Concatenate(axis=channel_axis, name=block_name + '_mixed')(branches)
        up = conv2d_bn(mixed, backend.int_shape(x)[channel_axis], 1, activation=None, use_bias=True,
                       name=block_name + '_conv', dropout_rate=dropout_rate, seed=seed)

        x = layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                          output_shape=backend.int_shape(x)[1:],
                          arguments={'scale': scale},
                          name=block_name)([x, up])
        if activation is not None:
            x = layers.Activation(activation, name=block_name + '_ac')(x)
        return x

    WEIGHTS_DIR = '/home/vmieuli/weights_files'
    WEIGHTS_FILE_NAME = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    img_input = layers.Input(shape=input_shape)

    # Stem block: 35 x 35 x 192
    x = conv2d_bn(img_input, 32, 3, strides=2, padding='valid', dropout_rate=-1, seed=seed)
    x = conv2d_bn(x, 32, 3, padding='valid', dropout_rate=-1, seed=seed)
    x = conv2d_bn(x, 64, 3, dropout_rate=-1, seed=seed)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = conv2d_bn(x, 80, 1, padding='valid', dropout_rate=-1, seed=seed)
    x = conv2d_bn(x, 192, 3, padding='valid', dropout_rate=-1, seed=seed)
    x = layers.MaxPooling2D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96, 1, dropout_rate=-1, seed=seed)
    branch_1 = conv2d_bn(x, 48, 1, dropout_rate=-1, seed=seed)
    branch_1 = conv2d_bn(branch_1, 64, 5, dropout_rate=-1, seed=seed)
    branch_2 = conv2d_bn(x, 64, 1, dropout_rate=-1, seed=seed)
    branch_2 = conv2d_bn(branch_2, 96, 3, dropout_rate=-1, seed=seed)
    branch_2 = conv2d_bn(branch_2, 96, 3, dropout_rate=-1, seed=seed)
    branch_pool = layers.AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, dropout_rate=-1, seed=seed)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    x = layers.Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x, scale=0.17, block_type='block35', block_idx=block_idx,
                                   dropout_rate=-1, seed=seed)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid', dropout_rate=dropout_rate, seed=seed)
    branch_1 = conv2d_bn(x, 256, 1, dropout_rate=dropout_rate, seed=seed)
    branch_1 = conv2d_bn(branch_1, 256, 3, dropout_rate=dropout_rate, seed=seed)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='valid', dropout_rate=dropout_rate, seed=seed)
    branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x, scale=0.1, block_type='block17', block_idx=block_idx,
                                   dropout_rate=dropout_rate, seed=seed)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1, dropout_rate=dropout_rate, seed=seed)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid', dropout_rate=dropout_rate, seed=seed)
    branch_1 = conv2d_bn(x, 256, 1, dropout_rate=dropout_rate, seed=seed)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='valid', dropout_rate=dropout_rate, seed=seed)
    branch_2 = conv2d_bn(x, 256, 1, dropout_rate=dropout_rate, seed=seed)
    branch_2 = conv2d_bn(branch_2, 288, 3, dropout_rate=dropout_rate, seed=seed)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='valid', dropout_rate=dropout_rate, seed=seed)
    branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x, scale=0.2, block_type='block8', block_idx=block_idx, dropout_rate=dropout_rate,
                                   seed=seed)
    x = inception_resnet_block(x, scale=1., activation=None, block_type='block8', block_idx=10,
                               dropout_rate=dropout_rate, seed=seed)

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 1536, 1, name='conv_7b', dropout_rate=dropout_rate, seed=seed)

    base_model = models.Model(img_input, x, name='inception_resnet_v2')
    if weights == 'imagenet':
        base_model.load_weights(os.path.join(WEIGHTS_DIR, WEIGHTS_FILE_NAME))

    # Classification block
    head_model = layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)
    head_model = layers.Dropout(rate=dropout_rate, seed=seed)(head_model, training=True)
    head_model = layers.Dense(classes, activation='softmax', name='predictions')(head_model)

    # Create complete model.
    return models.Model(img_input, head_model, name='inception_resnet_v2')'''


def bayesian_InceptionResNetV2(weights='imagenet', input_shape=(256, 256, 3),
                               classes=1000, dropout_rate=0.5, seed=1234):
    # code based on the one from:
    # https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_resnet_v2.py

    def conv2d_bn(x, filters, kernel_size, strides=1, padding='same', activation='relu',
                  use_bias=False, name=None, dropout_rate=0.5, seed=1234):
        if dropout_rate != -1:
            x = layers.Dropout(rate=dropout_rate, seed=seed)(x, training=True)
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, name=name)(x)
        if not use_bias:
            bn_axis = 1 if backend.image_data_format() == 'channels_first' else 3
            bn_name = None if name is None else name + '_bn'
            x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
            if activation is not None:
                ac_name = None if name is None else name + '_ac'
            x = layers.Activation(activation, name=ac_name)(x)
        return x

    def inception_resnet_block(x, scale, block_type, block_idx, activation='relu', dropout_rate=0.5, seed=1234):
        if block_type == 'block35':
            branch_0 = conv2d_bn(x, 32, 1, dropout_rate=dropout_rate, seed=seed)
            branch_1 = conv2d_bn(x, 32, 1, dropout_rate=dropout_rate, seed=seed)
            branch_1 = conv2d_bn(branch_1, 32, 3, dropout_rate=dropout_rate, seed=seed)
            branch_2 = conv2d_bn(x, 32, 1, dropout_rate=dropout_rate, seed=seed)
            branch_2 = conv2d_bn(branch_2, 48, 3, dropout_rate=dropout_rate, seed=seed)
            branch_2 = conv2d_bn(branch_2, 64, 3, dropout_rate=dropout_rate, seed=seed)
            branches = [branch_0, branch_1, branch_2]
        elif block_type == 'block17':
            branch_0 = conv2d_bn(x, 192, 1, dropout_rate=dropout_rate, seed=seed)
            branch_1 = conv2d_bn(x, 128, 1, dropout_rate=dropout_rate, seed=seed)
            branch_1 = conv2d_bn(branch_1, 160, [1, 7], dropout_rate=dropout_rate, seed=seed)
            branch_1 = conv2d_bn(branch_1, 192, [7, 1], dropout_rate=dropout_rate, seed=seed)
            branches = [branch_0, branch_1]
        elif block_type == 'block8':
            branch_0 = conv2d_bn(x, 192, 1, dropout_rate=dropout_rate, seed=seed)
            branch_1 = conv2d_bn(x, 192, 1, dropout_rate=dropout_rate, seed=seed)
            branch_1 = conv2d_bn(branch_1, 224, [1, 3], dropout_rate=dropout_rate, seed=seed)
            branch_1 = conv2d_bn(branch_1, 256, [3, 1], dropout_rate=dropout_rate, seed=seed)
            branches = [branch_0, branch_1]
        else:
            raise ValueError('Unknown Inception-ResNet block type. '
                             'Expects "block35", "block17" or "block8", '
                             'but got: ' + str(block_type))

        block_name = block_type + '_' + str(block_idx)
        channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
        mixed = layers.Concatenate(axis=channel_axis, name=block_name + '_mixed')(branches)
        up = conv2d_bn(mixed, backend.int_shape(x)[channel_axis], 1, activation=None, use_bias=True,
                       name=block_name + '_conv', dropout_rate=dropout_rate, seed=seed)

        x = layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                          output_shape=backend.int_shape(x)[1:],
                          arguments={'scale': scale},
                          name=block_name)([x, up])
        if activation is not None:
            x = layers.Activation(activation, name=block_name + '_ac')(x)
        return x

    WEIGHTS_DIR = '/home/vmieuli/weights_files'
    WEIGHTS_FILE_NAME = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    img_input = layers.Input(shape=input_shape)

    # Stem block: 35 x 35 x 192
    x = conv2d_bn(img_input, 32, 3, strides=2, padding='valid', dropout_rate=-1, seed=seed)
    x = conv2d_bn(x, 32, 3, padding='valid', dropout_rate=-1, seed=seed)
    x = conv2d_bn(x, 64, 3, dropout_rate=-1, seed=seed)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = conv2d_bn(x, 80, 1, padding='valid', dropout_rate=-1, seed=seed)
    x = conv2d_bn(x, 192, 3, padding='valid', dropout_rate=-1, seed=seed)
    x = layers.MaxPooling2D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96, 1, dropout_rate=-1, seed=seed)
    branch_1 = conv2d_bn(x, 48, 1, dropout_rate=-1, seed=seed)
    branch_1 = conv2d_bn(branch_1, 64, 5, dropout_rate=-1, seed=seed)
    branch_2 = conv2d_bn(x, 64, 1, dropout_rate=-1, seed=seed)
    branch_2 = conv2d_bn(branch_2, 96, 3, dropout_rate=-1, seed=seed)
    branch_2 = conv2d_bn(branch_2, 96, 3, dropout_rate=-1, seed=seed)
    branch_pool = layers.AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, dropout_rate=-1, seed=seed)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    x = layers.Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x, scale=0.17, block_type='block35', block_idx=block_idx,
                                   dropout_rate=-1, seed=seed)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid', dropout_rate=-1, seed=seed)
    branch_1 = conv2d_bn(x, 256, 1, dropout_rate=-1, seed=seed)
    branch_1 = conv2d_bn(branch_1, 256, 3, dropout_rate=-1, seed=seed)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='valid', dropout_rate=-1, seed=seed)
    branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x, scale=0.1, block_type='block17', block_idx=block_idx,
                                   dropout_rate=-1, seed=seed)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1, dropout_rate=dropout_rate, seed=seed)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid', dropout_rate=dropout_rate, seed=seed)
    branch_1 = conv2d_bn(x, 256, 1, dropout_rate=dropout_rate, seed=seed)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='valid', dropout_rate=dropout_rate, seed=seed)
    branch_2 = conv2d_bn(x, 256, 1, dropout_rate=dropout_rate, seed=seed)
    branch_2 = conv2d_bn(branch_2, 288, 3, dropout_rate=dropout_rate, seed=seed)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='valid', dropout_rate=dropout_rate, seed=seed)
    branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x, scale=0.2, block_type='block8', block_idx=block_idx, dropout_rate=dropout_rate,
                                   seed=seed)
    x = inception_resnet_block(x, scale=1., activation=None, block_type='block8', block_idx=10,
                               dropout_rate=dropout_rate, seed=seed)

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 1536, 1, name='conv_7b', dropout_rate=dropout_rate, seed=seed)

    base_model = models.Model(img_input, x, name='inception_resnet_v2')
    if weights == 'imagenet':
        base_model.load_weights(os.path.join(WEIGHTS_DIR, WEIGHTS_FILE_NAME))

    # Classification block
    head_model = layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)
    head_model = layers.Dropout(rate=dropout_rate, seed=seed)(head_model, training=True)
    head_model = layers.Dense(classes, activation='softmax', name='predictions')(head_model)

    # Create complete model.
    return models.Model(img_input, head_model, name='inception_resnet_v2')




def bayesian_ResNet50(weights='imagenet', input_shape=(256, 256, 3), classes=1000, dropout_rate=0.5, seed=222):
    def identity_block(input_tensor, kernel_size, filters, stage, block, dropout_rate=0.5, seed=1):
        filters1, filters2, filters3 = filters
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = layers.Dropout(rate=dropout_rate, seed=seed)(input_tensor, training=True)
        x = layers.Conv2D(filters1, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2a')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Dropout(rate=dropout_rate, seed=seed)(x, training=True)
        x = layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal',
                          name=conv_name_base + '2b')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Dropout(rate=dropout_rate, seed=seed)(x, training=True)
        x = layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        return layers.Activation('relu')(x)

    def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), dropout_rate=0.5, seed=1):
        filters1, filters2, filters3 = filters
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = layers.Dropout(rate=dropout_rate, seed=seed)(input_tensor, training=True)
        x = layers.Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='he_normal',
                          name=conv_name_base + '2a')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Dropout(rate=dropout_rate, seed=seed)(x, training=True)
        x = layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal',
                          name=conv_name_base + '2b')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Dropout(rate=dropout_rate, seed=seed)(x, training=True)
        x = layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = layers.Dropout(rate=dropout_rate, seed=1)(input_tensor, training=True)
        shortcut = layers.Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal',
                                 name=conv_name_base + '1')(shortcut)
        shortcut = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        return layers.Activation('relu')(x)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    WEIGHTS_DIR = '/home/vmieuli/weights_files'
    WEIGHTS_FILE_NAME = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    # Determine proper input shape
    img_input = layers.Input(shape=input_shape)
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Dropout(rate=dropout_rate, seed=seed)(x, training=True)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal', name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), dropout_rate=dropout_rate, seed=seed)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', dropout_rate=dropout_rate, seed=seed)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', dropout_rate=dropout_rate, seed=seed)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', dropout_rate=dropout_rate, seed=seed)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', dropout_rate=dropout_rate, seed=seed)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', dropout_rate=dropout_rate, seed=seed)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', dropout_rate=dropout_rate, seed=seed)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', dropout_rate=dropout_rate, seed=seed)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', dropout_rate=dropout_rate, seed=seed)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', dropout_rate=dropout_rate, seed=seed)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', dropout_rate=dropout_rate, seed=seed)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', dropout_rate=dropout_rate, seed=seed)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', dropout_rate=dropout_rate, seed=seed)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', dropout_rate=dropout_rate, seed=seed)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', dropout_rate=dropout_rate, seed=seed)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', dropout_rate=dropout_rate, seed=seed)

    base_model = models.Model(img_input, x, name='resnet50')
    '''if weights == 'imagenet':
        base_model.load_weights(os.path.join(WEIGHTS_DIR, WEIGHTS_FILE_NAME))'''

    # Classification block
    head_model = layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)
    head_model = layers.Dropout(rate=dropout_rate, seed=seed)(head_model)
    head_model = layers.Dense(classes, activation='softmax')(head_model)

    # Create complete model.
    return models.Model(img_input, head_model, name='resnet50')


def bayesian_ResNet18(input_shape, dropout_rate, seed, classes):
    '''
     code based on the one by:
     francesco.ponzio@polito.it
    '''

    def projection_shortcut(x, out_filters, stride, dropout_rate):
        x = keras.layers.Dropout(rate=dropout_rate, seed=seed)(x, training=True)
        return keras.layers.Convolution2D(out_filters, 1, strides=stride)(x)

    def resnet_block(x, filters, kernel, stride, dropout_rate):
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        if stride != 1 or filters != x.shape[1]:
            shortcut = projection_shortcut(x, filters, stride, dropout_rate)
        else:
            shortcut = x
        x = keras.layers.Dropout(rate=dropout_rate, seed=seed)(x, training=True)
        x = keras.layers.Convolution2D(filters, kernel, strides=stride, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Dropout(rate=dropout_rate, seed=seed)(x, training=True)
        x = keras.layers.Convolution2D(filters, kernel, strides=1, padding='same')(x)
        return keras.layers.add([x, shortcut])

    filters = [64, 128, 256, 512]
    kernels = [3, 3, 3, 3]
    strides = [1, 2, 2, 2]

    image = keras.layers.Input(shape=input_shape, dtype='float32')
    x = keras.layers.Dropout(rate=dropout_rate, seed=seed)(image, training=True)
    x = keras.layers.Convolution2D(64, 3, strides=1, padding='same')(x)

    for i in range(len(kernels)):
        x = resnet_block(x, filters[i], kernels[i], strides[i], dropout_rate)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(rate=dropout_rate, seed=seed)(x, training=True)
    x = keras.layers.AveragePooling2D(4, 1)(x)
    x = keras.layers.Dropout(rate=dropout_rate, seed=seed)(x, training=True)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(rate=dropout_rate, seed=seed)(x, training=True)
    x = keras.layers.Dense(classes, activation='softmax')(x)

    return keras.Model(inputs=image, outputs=x, name='resnet18')
