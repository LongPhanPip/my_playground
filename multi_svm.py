import numpy as np


def get_batch(X_train, y_train, batch_size=32):
    indexes = np.random.randint(0, len(X_train), size=batch_size)
    return X_train[indexes], y_train[indexs]



# Model
epochs = 10
learning_rate = 0.01
batch_size = 32
steps = len(X_train) // batch_size
classes = 20
gamma = 0.1
threshole = 1
weights = np.random.rand(classes, X_train.shape[-1])


def predict(X):
    z = weights.dot(X.T)
    return np.argmax(z)


# Traing
for i in range(epochs):
    for j in range(steps):
        X, y = get_batch(X_train, y_train)
        z = W.dot(X.T)
        z_max = z[y, list(np.range(len(X)))]
        loss = np.sum(np.where((threshole + z - z_max < 0) or (z == z_max), 0, threshole + z - z_max)) / len(X)
        gradient = np.where(loss > 0, 1, 0)
        gradient[y, list(np.range(len(X)))] = - (gradient == 1).sum(axis=0)

        d_weight = np.zeros(classes, X.shape[-1])
        for k in range(classes):
            d_weight[k, :] = np.sum(gradient * X.T, axis=1)

        regularizer = gamma * weights
        d_weight += regularizer

        weights -= d_weight
        print(loss)
