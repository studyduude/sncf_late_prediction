import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn.neighbors import KNeighborsRegressor

from preprocessing import load_and_process


def scaled_knn_error(data, departure_scaling, arrival_scaling):
    total_error = 0
    for x_train, y_train, x_test, y_test in data:
        x_train = x_train.copy()
        x_test = x_test.copy()
        x_train[:, 2:60] *= departure_scaling
        x_test[:, 2:60] *= departure_scaling
        x_train[:, 60:118] *= arrival_scaling
        x_test[:, 60:118] *= arrival_scaling
        dataset_error = []
        for n_neighbors in range(19, 76, 7):
            knn = KNeighborsRegressor(n_neighbors, n_jobs=-1)
            knn.fit(x_train, y_train)
            y_test_pred = knn.predict(x_test)
            error = np.sum((y_test_pred - y_test) ** 2)
            dataset_error.append(error)
        total_error += min(dataset_error) / x_test.shape[0]
    return total_error / len(data)


def main():
    data = [load_and_process()[:-2] for _ in range(3)]
    departure_scaling_1d = np.logspace(-0.5, 0.75, 15, base=10)
    arrival_scaling_1d = np.logspace(-0.5, 0.75, 15, base=10)
    departure_scaling, arrival_scaling = np.meshgrid(
        departure_scaling_1d, arrival_scaling_1d
    )
    errors = np.zeros_like(departure_scaling)
    for i, a_scaling in enumerate(tqdm.tqdm(arrival_scaling_1d)):
        for j, d_scaling in enumerate(departure_scaling_1d):
            error = scaled_knn_error(data, a_scaling, d_scaling)
            errors[i, j] = error
    plt.pcolor(departure_scaling, arrival_scaling, errors)
    plt.xscale("log")
    plt.yscale("log")
    plt.xticks(departure_scaling_1d, [f"{x:.2f}" for x in departure_scaling_1d])
    plt.yticks(departure_scaling_1d, [f"{y:.2f}" for y in arrival_scaling_1d])
    plt.xlabel("departure scaling")
    plt.ylabel("arrival scaling")
    plt.title("KNN average test error")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
