import logging
import logging.handlers
import tensorflow as tf
import keras.backend as K
from keras.models import model_from_json
from keras.callbacks import TensorBoard


def session_allow_growth():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    return sess


def keras_session():
    K.set_session(session_allow_growth())


def get_logger(logger_name, logger_path):
    logger = logging.getLogger(logger_name)

    formatter = logging.Formatter('[%(filename)s|%(asctime)s] %(message)s')

    # 스트림과 파일로 로그를 출력하는 핸들러를 각각 만든다.
    file_handler = logging.FileHandler(logger_path)
    stream_handler = logging.StreamHandler()

    # 각 핸들러에 포매터를 지정한다.
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # 로거 인스턴스에 스트림 핸들러와 파일핸들러를 붙인다.
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def keras_model_save_weights(model, path):
    model.save_weights('%s.h5' % path)


def keras_model_save_architecture(model, path):
    with open('%s.json' % path, 'w') as f:
        f.write(model.to_json())


def keras_model_load_weights(model, path):
    model.load_weights('%s.h5' % path)
    return model


def keras_model_load_architecture(path):
    with open('%s.json' % path, 'r') as f:
        model = model_from_json(f.read())

    return model


def get_tb(path):
    return TensorBoard(path, histogram_freq=1)
