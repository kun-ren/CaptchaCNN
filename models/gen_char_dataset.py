import numpy as np

from input import InputFlow
from dataset import CaptchaDataset
from chars import find_chars

'''
This script will generate a dataset which consists of individual character
images (extracted from captcha dataset samples) and their corresponding label
'''


# Image dimensions for each sample: height x width
IMAGE_SIZE = (45, 40)


class CharImageGenerator(InputFlow):
    '''
    Exposes the iterator interface and yields the captcha characters and labels
    '''
    def __init__(self, m_dataset):
        self.dataset = m_dataset
        super().__init__(
            self.dataset.X,
            self.dataset.y,
            batch_size=1,
            shuffle=False,      # 非随机：保证顺序不重复
            generate_samples=0  # 不额外扩增
        )

    def __iter__(self):
        text_size = self.dataset.text_size

        it = super().__iter__()
        while True:
            X_batch, y_batch = next(it)

            chars = find_chars(
                X_batch[0, :, :, 0],
                char_size=IMAGE_SIZE,
                num_chars=text_size
            )

            for k in range(0, text_size):
                yield chars[k], y_batch[0, k, :]


if __name__ == '__main__':
    import pandas as pd

    dataset = CaptchaDataset("../../dataset")

    X_list = []
    y_list = []
    generator = iter(CharImageGenerator(dataset))

    for X_char, y_char in generator:
        X_list.append(X_char)
        y_list.append(y_char)

        if len(X_list) == dataset.num_samples * dataset.text_size:
            break

    X = np.array(X_list).astype(np.float32).reshape((-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    y = np.array(y_list).astype(np.uint8)

    df = pd.DataFrame.from_dict({
        'attr': ['Number of samples', 'Image dimensions', 'Number of char classes'],
        'values': [X.shape[0], X.shape[1:], y.shape[1]]
    })

    print(y.shape)
    df.set_index('attr', inplace=True)
    print(df)

    # Save final dataset
    np.savez_compressed('.chars.npz', X=X, y=y)
