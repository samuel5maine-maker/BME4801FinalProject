import numpy as np
import time

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit


# ---------------------------------------------------------------------------
# Label names
# ---------------------------------------------------------------------------

label_names = {
    0: "Normal",
    1: "Choroidal neovascularization",
    2: "Diabetic macular edema",
    3: "Drusen"
}


# ---------------------------------------------------------------------------
# Preprocessing functions
# ---------------------------------------------------------------------------

def avg_2x2_pool(array):
    '''Reduced dimensionality of a 3d square array by half
    Takes the average of neighboring squares to find value'''

    x, y, z = array.shape

    reduced_array = np.zeros((x, int(y/2), int(z/2)))
    for i in range(x):
        for j in range(0, y, 2):
            for k in range(0, z, 2):
                p0 = float(array[i][j][k])
                p1 = float(array[i][j][k+1])
                p2 = float(array[i][j+1][k])
                p3 = float(array[i][j+1][k+1])

                new_point = (p0 + p1 + p2 + p3)/4.0

                reduced_array[i][int(j/2)][int(k/2)] = new_point

    return reduced_array


def max_2x2_pool(array):
    '''Reduces the dimensions of a 3D square array by half
     takes the max of neighboring squares to determine the value '''
    x, y, z = array.shape

    reduced_array = np.zeros((x, y//2, z//2), dtype=np.float32)

    for i in range(x):
        for j in range(0, y, 2):
            for k in range(0, z, 2):
                p0 = float(array[i][j][k])
                p1 = float(array[i][j][k+1])
                p2 = float(array[i][j+1][k])
                p3 = float(array[i][j+1][k+1])

                reduced_array[i][j//2][k//2] = max(p0, p1, p2, p3)

    return reduced_array


def median_filter_2d(img2d, size=3):
    '''Takes in a 2d array and looks at nearby pixels in radius of size
    it will then sort those pixels and replace pixel at that position of arr
    with the median value'''

    x, y = img2d.shape
    r = size//2

    arr = np.zeros((x, y), dtype=float)

    for i in range(x):
        for j in range(y):
            window = []
            for ri in range(-r, r + 1):
                for rj in range(-r, r + 1):
                    ii = i + ri
                    jj = j + rj

                    if ii < 0:
                        ii = 0
                    if ii >= x:
                        ii = x - 1
                    if jj < 0:
                        jj = 0
                    if jj >= y:
                        jj = y - 1

                    window.append(float(img2d[ii][jj]))
            window.sort()
            arr[i][j] = window[len(window)//2]
    return arr


def median(array, size=3):
    x, y, z = array.shape
    filtered = np.zeros((x, y, z), dtype=float)

    for i in range(x):
        filtered[i] = median_filter_2d(array[i].astype(np.float32), size=size)

    return filtered


def flatten_data(X):
    '''Flattens a 3D array into a 2D one'''
    new_arr = X.reshape(X.shape[0], -1)
    return new_arr


def basic_new_features(images):
    x, y, z = images.shape
    features = np.zeros((x, 7), dtype=float)

    for i in range(x):
        flat = images[i].reshape(-1).astype(np.float32)

        mean = float(flat.mean())
        std  = float(flat.std())
        mn   = float(flat.min())
        mx   = float(flat.max())
        med  = float(np.median(flat))
        q1   = float(np.percentile(flat, 25))
        q3   = float(np.percentile(flat, 75))

        features[i, 0] = mean
        features[i, 1] = std
        features[i, 2] = mn
        features[i, 3] = mx
        features[i, 4] = med
        features[i, 5] = q1
        features[i, 6] = q3

    return features


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

flattener = FunctionTransformer(flatten_data, validate=False)
scalar = StandardScaler()
svc = LinearSVC()

def build_pipelines():
    '''Returns a dict of named sklearn Pipelines ready to fit'''
    pipelines = {
        "Baseline (784px)": Pipeline([
            ("flat", flattener),
            ("scale", MinMaxScaler()),
            ("model", LinearSVC(
                dual=True,
                max_iter=1000
            ))
        ]),

        "Avg pool (196px)": Pipeline([
            ("avg14", FunctionTransformer(avg_2x2_pool, validate=False)),
            ("flat", FunctionTransformer(flatten_data, validate=False)),
            ("scale", MinMaxScaler()),
            ("model", LinearSVC(
                dual=True,
                max_iter=1000
            ))
        ]),

        "Avg pool (49px)": Pipeline([
            ("avg14", FunctionTransformer(avg_2x2_pool, validate=False)),
            ("avg7",  FunctionTransformer(avg_2x2_pool, validate=False)),
            ("flat",  FunctionTransformer(flatten_data, validate=False)),
            ("scale", MinMaxScaler()),
            ("model", LinearSVC(
                dual=True,
                max_iter=1000
            ))
        ]),

        "Median filter then avg pool (196px)": Pipeline([
            ("med",   FunctionTransformer(median, validate=False)),
            ("avg14", FunctionTransformer(avg_2x2_pool, validate=False)),
            ("flat",  FunctionTransformer(flatten_data, validate=False)),
            ("scale", MinMaxScaler()),
            ("model", LinearSVC(
                dual=True,
                max_iter=1000
            ))
        ]),

        "Max pool (196px)": Pipeline([
            ("max14", FunctionTransformer(max_2x2_pool, validate=False)),
            ("flat",  FunctionTransformer(flatten_data, validate=False)),
            ("scale", MinMaxScaler()),
            ("model", LinearSVC(
                dual=True,
                max_iter=1000
            ))
        ]),

        "PCA 50": Pipeline([
            ("flat",  FunctionTransformer(flatten_data, validate=False)),
            ("scale", MinMaxScaler()),
            ("pca",   PCA(n_components=50)),
            ("model", LinearSVC(
                dual=True,
                max_iter=1000
            ))
        ]),

        "Generated features": Pipeline([
            ("features", FunctionTransformer(basic_new_features, validate=False)),
            ("scale",    MinMaxScaler()),
            ("model",    LinearSVC(
                dual=True,
                max_iter=1000
            ))
        ]),

        "Median then PCA": Pipeline([
            ("med",   FunctionTransformer(median, validate=False)),
            ("flat",  FunctionTransformer(flatten_data, validate=False)),
            ("scale", MinMaxScaler()),
            ("pca",   PCA(n_components=200)),
            ("model", LinearSVC(
                dual=True,
                max_iter=1000
            ))
        ]),
    }

    return pipelines


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_eval_model(name, model, x_tr, y_tr, x_va, y_va):
    '''trains and evaluates the model'''
    t1 = time.time()
    model.fit(x_tr, y_tr)
    t2 = time.time()

    pred = model.predict(x_va)

    acc    = accuracy_score(y_va, pred)
    prec   = precision_score(y_va, pred, average="weighted", zero_division=1)
    recall = recall_score(y_va, pred, average="weighted", zero_division=1)

    result = {
        "Model": name,
        "Train Time (s)": round(t2-t1, 3),
        "Val Acc": round(acc, 4),
        "Val Precision": round(prec, 4),
        "Val Recall": round(recall, 4)
    }

    print(f"Model: {name}")
    print(f"Training Time: {t2-t1:.3f}")
    print(f'acc: {acc:.4f}')
    print(f'prec:{prec:.4f}')
    print(f'recall:{recall:.4f}\n')

    return result


def stratified_subset(X, y, n_samples):
    n_samples = min(n_samples, X.shape[0])
    split = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=42)
    idx, _ = next(split.split(X, y))
    return X[idx], y[idx]
