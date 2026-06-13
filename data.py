import struct
import numpy as np

class MNISTLoader:
    def __init__(self, images_path, labels_path):
        self.images_path = images_path
        self.labels_path = labels_path

    def load(self, normalize=True):
        # Labels
        with open(self.labels_path, "rb") as f:
            magic, n_labels = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(f"Expected label magic 2049, got {magic}")

            labels = np.frombuffer(f.read(), dtype=np.uint8)

        # Images
        with open(self.images_path, "rb") as f:
            magic, n_images, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(f"Expected image magic 2051, got {magic}")

            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(n_images, rows * cols)

        if n_images != n_labels:
            raise ValueError(
                f"Image count ({n_images}) != Label count ({n_labels})"
            )

        if normalize:
            images = images.astype(np.float32) / 255.0

        return images, labels

train_loader = MNISTLoader(
    r"train-images-idx3-ubyte",
    r"train-labels-idx1-ubyte"
)

test_loader = MNISTLoader(
    r"t10k-images-idx3-ubyte",
    r"t10k-labels.idx1-ubyte"
)

def get_data():
    X_train, y_train = train_loader.load()
    X_test, y_test = test_loader.load()
    return X_train, y_train, X_test, y_test

