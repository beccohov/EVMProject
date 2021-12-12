import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


class RandomForestMSE:
    def __init__(
            self, n_estimators, max_depth=None, feature_subsample_size=None,
            **trees_parameters
    ):
        if feature_subsample_size:
            self.feature_subsample_size = feature_subsample_size
        else:
            self.feature_subsample_size = 1 / 3
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.tree_params = trees_parameters
        self.ensemble = []
        self.history = {'val_score': [], 'train_score': []}

    def fit(self, X, y, X_val=None, y_val=None):
        N = X.shape[0]
        self.subfeatures = np.zeros((self.n_estimators, int(X.shape[1] * self.feature_subsample_size)), dtype='int')
        if X_val is not None:
            y_cur = np.zeros(X_val.shape[0])
        y_train_cur = np.zeros(X.shape[0])
        for estimator_order in range(self.n_estimators):
            bagged_elements = np.random.randint(0, N, N)
            bagged_features = np.random.randint(0, X.shape[1], int(X.shape[1] * self.feature_subsample_size))
            self.subfeatures[estimator_order, :] = bagged_features
            x_train = X[bagged_elements, :][:, bagged_features]
            y_train = y[bagged_elements]
            estimator = DecisionTreeRegressor(max_depth=self.max_depth, criterion="squared_error",
                                              **self.tree_params)  # MSE used as proposed
            estimator.fit(x_train, y_train)
            y_train_cur += estimator.predict(X[:, bagged_features])
            self.history['train_score'].append(
                mean_squared_error(y_train, y_train_cur / (estimator_order + 1), squared=False))
            if X_val is not None:
                y_cur += estimator.predict(X_val[:, bagged_features])
                self.history['val_score'].append(
                    mean_squared_error(y_val, y_cur / (estimator_order + 1), squared=False))
            self.ensemble.append(estimator)

    def predict(self, X):
        if not self.ensemble:
            raise NotImplementedError  # unfitted yet
        N_acessible = len(self.ensemble)
        predictions = np.zeros(X.shape[0])
        for i, estimator in enumerate(self.ensemble):
            predictions += estimator.predict(X[:, self.subfeatures[i]])
        return predictions / N_acessible


class GradientBoostingMSE:
    def __init__(
            self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
            **trees_parameters
    ):
        if feature_subsample_size:
            self.feature_subsample_size = feature_subsample_size
        else:
            self.feature_subsample_size = 1 / 3
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.treeparams = trees_parameters
        self.history = {'val_score': [], 'train_score': []}

    def fit(self, X, y, X_val=None, y_val=None):
        self.subfeatures = np.zeros((self.n_estimators, int(X.shape[1] * self.feature_subsample_size)), dtype='int')
        bagged_features = np.random.randint(0, X.shape[1], int(X.shape[1] * self.feature_subsample_size))
        self.subfeatures[0, :] = bagged_features
        self.forest = [DecisionTreeRegressor(max_depth=self.max_depth, **self.treeparams)]
        self.coefs = [1]
        self.forest[0].fit(X[:, bagged_features], y)
        y_pred_cur = self.predict(X)
        self.history['train_score'].append(mean_squared_error(y, y_pred_cur, squared=False))
        if not X_val is None:
            y_pred_val = self.predict(X_val)
            self.history['val_score'].append(mean_squared_error(y_val, y_pred_val, squared=False))

        for estimator_order in range(self.n_estimators - 1):
            bagged_features = np.random.randint(0, X.shape[1], int(X.shape[1] * self.feature_subsample_size))
            self.subfeatures[estimator_order + 1, :] = bagged_features
            antigrad = 2 * (y - y_pred_cur)
            new_estimator = DecisionTreeRegressor(max_depth=self.max_depth, **self.treeparams)
            new_estimator.fit(X[:, bagged_features], antigrad)

            self.forest.append(new_estimator)
            b_n = new_estimator.predict(X[:, bagged_features])
            newgamma = minimize_scalar(lambda x: mean_squared_error(y, y_pred_cur + x * b_n)).x
            self.coefs.append(newgamma * self.lr)
            y_pred_cur += self.lr * newgamma * b_n
            self.history['train_score'].append(mean_squared_error(y, y_pred_cur, squared=False))
            if not X_val is None:
                y_pred_val += self.lr * newgamma * new_estimator.predict(X_val[:, bagged_features])
                self.history['val_score'].append(mean_squared_error(y_val, y_pred_val, squared=False))

    def predict(self, X):
        gammas = np.array(self.coefs)
        pure_preds = np.zeros((X.shape[0], gammas.shape[0]))
        for i, estimator in enumerate(self.forest):
            pure_preds[:, i] = estimator.predict(X[:, self.subfeatures[i]])
        return pure_preds.dot(gammas.T).reshape(-1, )

