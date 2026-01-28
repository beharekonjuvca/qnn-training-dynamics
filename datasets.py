import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_moons(n_samples=200, noise=0.1, test_size=0.2, seed=42):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=test_size, random_state=seed)


def load_circles(n_samples=200, noise=0.1, factor=0.5, test_size=0.2, seed=42):
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=seed
    )

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=test_size, random_state=seed)
