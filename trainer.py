from utils import get_logger, get_tb, keras_model_save_weights, keras_model_save_architecture,\
    keras_model_load_architecture, keras_model_load_weights
from keras.optimizers import Adam,sgd
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from Utils.data_utils import load_CIFAR10
import numpy as np

class Trainer:

    def __init__(self,config,model):
        self.config = config
        self.model = model

        self.data_path = self.config.config['data']['path']['data_path']
        self.save_path = self.config.config['result']['path']['save_path']
        self.log_path = self.config.config['result']['path']['log_path']
        self.tb_path = self.config.config['result']['path']['tb_path']
        self.archi_path = self.config.config['result']['path']['archi_path']


        self.name = self.config.config['general']['name']
        self.nb_classes = self.config.config['ingredient']['param']['nb_classes']
        self.batch_size  =self.config.config['train_param']['batch_size']
        self.lr = self.config.config['train_param']['learning_rate']
        self.epochs = self.config.config['train_param']['epochs']
        self.decay = self.config.config['train_param']['decay']
        self.es_tolerance = self.config.config['train_param']['es_tolerance']
        self.key_metric = self.config.config['train_param']['key_metric']

        self.tb = get_tb(path=self.tb_path)
        self.logger = get_logger(self.name, self.log_path)

    def load_data(self):
        X_train, Y_train, X_val, Y_val, X_test, Y_test, Class_names = load_CIFAR10()

        #one-hot encoding
        Y_train = to_categorical(Y_train,self.nb_classes)
        Y_val = to_categorical(Y_val,self.nb_classes)
        Y_test = to_categorical(Y_test,self.nb_classes)

        X_train = np.array(np.concatenate((X_train,X_val),axis=0))
        Y_train = np.array(np.concatenate((Y_train,Y_val),axis=0))


        return {'X_train': X_train, 'Y_train': Y_train,
                'X_test': X_test, 'Y_test': Y_test}

    def _set_optimizer(self, last_epoch):
        lr = self.lr * ((1 - self.decay) ** last_epoch)
        decay = self.decay
        adam = Adam(lr=lr, decay=decay)
        return adam

    def _log(self, epoch, history):
        train_loss = history.history['loss'][0]

        # train
        self.logger.warning('[%s|Train|Epoch %d] Loss: %f' % (self.name, epoch, train_loss))

        # eval
        eval_results = {'Loss': history.history['val_loss'][0],
                        'Accuracy': history.history['val_acc'][0]}
        self.logger.warning('####################')
        self.logger.warning('[%s|Valid|Epoch %d]' % (self.name, epoch))
        for k, v in eval_results.items():
            self.logger.warning('%s: %f' % (k, v))
        self.logger.warning('####################')

    def train(self):

        model = self.model
        keras_model_save_architecture(model, self.archi_path)
        last_epoch = 0


        #multi-GPU
        #if self.config.multi_gpu > 1:
        #    model = multi_gpu_model(model, gpus=self.config.config.multi_gpu)

        optimizer = self._set_optimizer(last_epoch=last_epoch)
        model.compile(optimizer=optimizer,
                      loss=categorical_crossentropy,
                      metrics=['accuracy'])

        es = self.es_tolerance
        best_eval = 100001
        _eval = 100000
        for epoch in range(last_epoch, 100000):
            gens = self.load_data()

            history = model.fit(gens['X_train'],gens['Y_train'],
                                batch_size = self.batch_size, epochs=1, verbose=1,
                                validation_split=0.2,
                                shuffle=True)

            eval_results = {'Loss': history.history['val_loss'][0],
                            'Accuracy': history.history['val_acc'][0]}

            self._log(epoch=epoch, history=history)

            # save
            _eval = eval_results[self.key_metric]
            if _eval < best_eval:
                es = self.es_tolerance
                keras_model_save_weights(model, path=self.save_path + '%d' % epoch)
                best_eval = _eval
            else:
                if es > 0:
                    es -= 1
                    print('tolerance',es)
                else:
                    break
