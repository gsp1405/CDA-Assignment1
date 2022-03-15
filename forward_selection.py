import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


def forward_selection(X, y, num_of_features=1):
    features = np.array([])
    kf = KFold(n_splits=5)
    for i in range(num_of_features):
        AIC_best = np.finfo(float).min
        for feature in range(X.shape[1]):
            if feature in features:
                continue
            feature_cand = np.append(features, feature)
            model = LinearRegression()
            X_select = X.iloc[:, feature_cand]
            sse = np.zeros(5)
            for k, (train_index, test_index) in enumerate(kf.split(X_select)):
                X_train, X_test = X_select.iloc[train_index, :], X_select.iloc[test_index, :]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model.fit(X_train, y_train)
                y_hat = model.predict(X_test)
                sse[k] = np.sum((y_hat - y_test)**2)
            AIC = 2 * feature_cand.size - 2 * np.log(np.mean(sse))
            if AIC > AIC_best:
                AIC_best = AIC
                best_feature = feature
        features = np.append(features, best_feature)
    return features.astype(int)
