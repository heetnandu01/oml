import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def load_iris_data(test_size=0.3, random_state=42):
    """Load Iris dataset and return train-validation split."""
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_val, y_train, y_val


def decode_position(position):
    """Convert particle position into SVC hyperparameters."""
    log10_c = position[0]
    log10_gamma = position[1]
    c_value = 10 ** log10_c
    gamma_value = 10 ** log10_gamma
    return c_value, gamma_value


def pso_objective_factory(X_train, X_val, y_train, y_val):
    """
    Build a vectorized objective function for pyswarms.

    Each particle is interpreted as:
    position[0] = log10(C)
    position[1] = log10(gamma)

    Objective: minimize (1 - validation_accuracy).
    """

    def objective(positions):
        costs = []
        for position in positions:
            c_value, gamma_value = decode_position(position)
            model = SVC(C=c_value, gamma=gamma_value, kernel="rbf", random_state=42)
            try:
                model.fit(X_train, y_train)
                accuracy = model.score(X_val, y_val)
                cost = 1.0 - accuracy
            except Exception:
                # Fallback to high cost for any numerical/training issue.
                cost = 1.0
            costs.append(cost)

        return np.array(costs)

    return objective
