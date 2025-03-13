from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Input, AlphaDropout, MultiHeadAttention, LayerNormalization, Lambda
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Nadam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow as tf


def create_model_0(input_dim, num_classes, learning_rate):
    """Modelo DNN simples com 2 camadas ocultas."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        Dropout(0.6),
        Dense(128, activation='relu'),
        Dropout(0.6),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    return model, optimizer


def create_model_1(input_dim, num_classes, learning_rate):
    """Modelo DNN com BatchNormalization e LeakyReLU."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.5),
        Dense(128),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    return model, optimizer


def create_model_2(input_dim, num_classes, learning_rate):
    """Modelo DNN com LeakyReLU e Dropout."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512),
        LeakyReLU(),
        Dropout(0.3),
        Dense(256),
        LeakyReLU(),
        Dropout(0.3),
        Dense(128),
        LeakyReLU(),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    return model, optimizer


def create_model_3(input_dim, num_classes, learning_rate):
    """Modelo DNN com RMSprop e ExponentialDecay."""
    lr_schedule = ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=10000, decay_rate=0.9)
    optimizer = RMSprop(learning_rate=lr_schedule)
    
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    return model, optimizer


def create_model_4(input_dim, num_classes, learning_rate):
    """Modelo DNN com SGD e ExponentialDecay para ajustar o learning rate."""
    lr_schedule = ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=10000, decay_rate=0.9)
    optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)
    
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    return model, optimizer


def create_model_5(input_dim, num_classes, learning_rate):
    """Modelo DNN com BatchNormalization, LeakyReLU e Dropout."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.25),
        Dense(384),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.25),
        Dense(256),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.25),
        Dense(128),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.25),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    return model, optimizer


# Novos Modelos (Substituindo TensorFlow Addons)

def create_model_6(input_dim, num_classes, learning_rate):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, activation='selu'),
        AlphaDropout(0.2),
        Dense(256, activation='selu'),
        AlphaDropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    return model, optimizer


def create_model_7(input_dim, num_classes, learning_rate):
    """Modelo DNN com Swish e Nadam como alternativa ao Ranger."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, activation='swish'),
        Dropout(0.3),
        Dense(256, activation='swish'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = Nadam(learning_rate=learning_rate)  # Substitui Ranger por Nadam
    return model, optimizer


def create_model_8(input_dim, num_classes, learning_rate):
    """Modelo DNN com Feature-wise Batch Normalization (FiLM)."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512),
        BatchNormalization(),
        Lambda(lambda x: x * 0.5),  # FiLM simples
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    return model, optimizer


def create_model_9(input_dim, num_classes, learning_rate):
    """Modelo DNN com Transformer Encoder para modelar dependências temporais."""
    inputs = Input(shape=(input_dim,))
    x = Dense(512, activation='relu')(inputs)
    x = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = LayerNormalization()(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, x)
    optimizer = Adam(learning_rate=learning_rate)
    return model, optimizer
