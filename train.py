from config import Config
from utils import keras_session
from trainer import Trainer
from densenet import DenseNet



def main():

    config_path = '/home2/jypark/CIFAR1000/config.yaml'

    keras_session()
    config = Config(config_path)

    model = DenseNet(config=config)
    trainer = Trainer(config=config,
                      model=model())

    trainer.train()

if __name__ == '__main__':
    main()