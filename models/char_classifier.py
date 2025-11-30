
import numpy as np
import keras
from keras.layers import *

from dataset import CaptchaDataset
from gen_char_dataset import CharImageGenerator


def generate_dataset(path):
    X_list = []
    y_list = []

    dataset = CaptchaDataset(path)
    dataset_generator = iter(CharImageGenerator(dataset))

    for X_char, y_char in dataset_generator:
        X_list.append(X_char)
        y_list.append(y_char)

        if len(X_list) == dataset.num_samples * dataset.text_size:
            break

    X = np.array(X_list).astype(np.float32).reshape((-1, 45, 40, 1))
    y = np.array(y_list).astype(np.uint8)
    print(X.shape)
    print(y.shape)

    return X, y

class CharClassifier(keras.models.Model):
    '''
    This class defines a convolutional neuronal network to classify individual
    characters on captcha image
    '''
    def __init__(self, char_size=(45, 40)):
        num_classes = 10

        # The next lines defines the layers of the CNN

        t_in = Input(shape=char_size + (1,), dtype=np.float32)

        x = t_in

        x = Conv2D(32, kernel_size=(5, 5), kernel_initializer='he_normal', padding='same')(x)
        x = MaxPool2D((2, 2))(x)

        x = Conv2D(64, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(x)
        x = MaxPool2D((2, 2))(x)

        x = Conv2D(32, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(x)
        x = MaxPool2D((2, 2))(x)

        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(num_classes, activation='softmax')(x)

        t_out = x

        # Initialize super instance (a keras model)
        super().__init__([t_in], [t_out])

        # Compile the model
        self.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def load_weights(self):
        '''
        Load the weights obtained at the training phase of this model by previous executions
        of this script. If no weights were generated previously, this method dont do anything.
        '''
        try:
            super().load_weights('.char-classifier-weights.hdf5')
        except:
            pass

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    import matplotlib.pyplot as plt
    import seaborn as sns
    from argparse import ArgumentParser

    # Process command line arguments
    parser = ArgumentParser(description='CLI to train/evaluate char classifier')
    parser.add_argument('-t', '--train', action='store_true', default=False, help='Train the model')
    parser.add_argument('-e', '--eval', action='store_true', default=False, help='Evaluate the model')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Enable verbosity mode')
    parser.add_argument('--batch-size', type=int, nargs=1, default=[8], help='Batch size when training the model')
    parser.add_argument('--epochs', type=int, nargs=1, default=[4], help='Max number of epochs of training phase')
    parser.add_argument('--test-size', type=float, nargs=1, default=[0.15], help='Test set size ratio in the range (0,1). Only used when test is enabled')
    parser.add_argument('--num-samples', type=int, nargs=1, help='Number of samples to be used to train/eval the model instead of the whole dataset')

    params = parser.parse_args()

    X_train, y_train = generate_dataset("../dataset")
    n = y_train.shape[0]

    train, eval, verbose = params.train, params.eval, params.verbose
    batch_size, epochs, test_size = params.batch_size[0], params.epochs[0],params.test_size[0]
    num_samples = min(params.num_samples[0], n) if params.num_samples is not None else n

    if num_samples <= 0:
        parser.error('Num samples must be a number greater than 0')

    if not train and not eval:
        parser.error('Either train or eval parameter must be set to true')

    if num_samples < n:
        # Use only part of the dataset
        indices = np.random.choice(np.arange(0, n), size=num_samples, replace=False)
        X_train, y_train = X_train[indices], y_train[indices]

    # Get char labels
    y_labels = y_train.argmax(axis=1)

    # Build the model
    model = CharClassifier(X_train.shape[1:3])

    # Show model info
    if verbose:
        model.summary()

    if train:
        if verbose:
            print('Training the model...')
        # Train the model
        result = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=test_size,)
        history = result.history
        model.save_weights(".char-classifier-weights.hdf5")

        # Show train performance score history

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))

        plt.sca(ax[0])
        plt.plot(history['loss'], color='red')
        plt.plot(history['val_loss'], color='blue')
        plt.legend(['Loss', 'Val. Loss'])
        plt.xlabel('Epoch')
        plt.title('Loss')
        plt.tight_layout()

        plt.sca(ax[1])
        plt.plot(history['acc'], color='red')
        plt.plot(history['val_acc'], color='blue')
        plt.legend(['Accuracy', 'Val. Accuracy'])
        plt.xlabel('Epoch')
        plt.title('Accuracy')

        plt.suptitle('Model performance on training')

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

    elif eval:
        # Load previously computed weights if train is off and eval enabled
        model.load_weights()


    if eval:
        if verbose:
            print('Testing the model...')

        X_test, y_test = generate_dataset("../dataset_predict")

        # Evaluate the model on test
        y_test_pred = model.predict(X_test, verbose=verbose)

        # Show accuracy score
        y_test_labels = y_test.argmax(axis=1)
        y_test_labels_pred = y_test_pred.argmax(axis=1)

        num_chars_per_captcha = 4
        alphabet = list("0123456789")

        # 将索引映射为字符
        chars = [alphabet[idx] for idx in y_test_labels_pred]

        # 每 4 个字符组成一个验证码字符串
        captchas = [''.join(chars[i:i + num_chars_per_captcha])
                    for i in range(0, len(chars), num_chars_per_captcha)]

        print(captchas)

        print('Accuracy on test set: {}'.format(np.round(accuracy_score(y_test_labels, y_test_labels_pred), 4)))

        # Show evaluation confusion matrx
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_test_labels, y_test_labels_pred), annot=True, fmt='d',
                                    xticklabels=alphabet, yticklabels=alphabet)
        plt.title('Confusion matrix of eval predictions')


    plt.show()
