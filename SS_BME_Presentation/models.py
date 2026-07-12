#Compilation file containing built models from each milestone
#Used for easy imports as well as easy viewing
from functools import partial
import numpy as np
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import StratifiedShuffleSplit

'''
!!!!!
UTILITY FUNCTIONS
!!!!
'''
#moved here for convience but also in utils.py on github
#This way if you want to run the notebook on your local machine you have everything in this file

def stratified_subset(X, y, n_samples):
    n_samples = min(n_samples, X.shape[0])
    split = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=42)
    idx, _ = next(split.split(X, y))
    return X[idx], y[idx]

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


def prep_for_cnn(X):
    X = X.astype("float32") / 255.0
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    return X


flattener = FunctionTransformer(flatten_data, validate=False)



#Helper callback necessary for implementation of wide and deep preliminary design
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                     restore_best_weights=True)

'''
Each Preliminary Design has been wrapped in a build function so can be dropped into a notebook
And used if you wish :)


Most preliminary designs were taken straight from their respective milestone.
Some were updated slightly 
'''

#preprocessed stacking preliminary design
def build_preprocessed_stacking():
    '''
    Preproceessed Stacking Classifier
    Preprocessing:
    Uses median filtering then average 2x2 scaling for both
    For SVC Scales Inputs and uses Nystroem approximation
    Models:
    SVC
    RF

    finally class weights were added as per professor recommendation
    '''
    rf_pipe = Pipeline([
        ("med",  FunctionTransformer(median, validate=False)),
        ("avg",  FunctionTransformer(avg_2x2_pool, validate=False)),
        ("flat", FunctionTransformer(flatten_data, validate=False)),
        ("rf",   RandomForestClassifier(
            max_depth=None,
            max_features=0.25,
            min_samples_leaf=1,
            min_samples_split=10,
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    svc_pipe = Pipeline([
        ("med",    FunctionTransformer(median, validate=False)),
        ("avg",    FunctionTransformer(avg_2x2_pool, validate=False)),
        ("flat",   FunctionTransformer(flatten_data, validate=False)),
        ("scale",  MinMaxScaler()),
        ("kernel", Nystroem(kernel="rbf", gamma=None, n_components=500, random_state=42)),
        ("svc",    LinearSVC(dual=False, max_iter=4000, class_weight='balanced')),
    ])

    return StackingClassifier(
        estimators=[
            ("rf",  rf_pipe),
            ("svc", svc_pipe),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=2,
        verbose=2,
    )


def build_preprocessed_stacking_tunable():
    rf_pipe = Pipeline([
        ("rf", RandomForestClassifier(
            max_depth=None,
            max_features=0.25,
            min_samples_leaf=1,
            min_samples_split=10,
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    svc_pipe = Pipeline([
        ("scale",  MinMaxScaler()),
        ("kernel", Nystroem(kernel="rbf", gamma=None, n_components=500, random_state=42)),
        ("svc",    LinearSVC(dual=False, max_iter=4000, class_weight='balanced')),
    ])

    return StackingClassifier(
        estimators=[
            ("rf",  rf_pipe),
            ("svc", svc_pipe),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=2,
        verbose=2,
    )


#Soft Voting Classifier Preliminary Design
def build_voting_classifier():
    """
    Soft-voting ensemble. Our soft voting classifier
    Preprocessing: flatten only.
    Models:
    RandomForestClassifier
    SVC
    AdaBoostClassifier

    Once again class weights added    
    """
    rf = Pipeline([
        ("flat",  FunctionTransformer(flatten_data, validate=False)),
        ("model", RandomForestClassifier(
            max_depth=None,
            max_features=0.25,
            min_samples_leaf=1,
            min_samples_split=10,
            n_estimators=200,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1,
        )),
    ])

    svc = Pipeline([
        ("flat",  FunctionTransformer(flatten_data, validate=False)),
        ("scale", MinMaxScaler()),
        ("model", SVC(kernel="rbf", probability=True, random_state=42, class_weight='balanced')),
    ])

    ada = Pipeline([
        ("flat",  FunctionTransformer(flatten_data, validate=False)),
        ("model", AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=200,
            learning_rate=1.0,
            random_state=42,
        )),
    ])

    return VotingClassifier(
        estimators=[
            ("rf",  rf),
            ("svc", svc),
            ("ada", ada),
        ],
        verbose=True,
        n_jobs=-1,
        voting="soft",
    )


#SVD Preliminary Design
def build_svd_soft_voting(n_svd_components= 100):
    """
    Soft-voting ensemble with MinMaxScaler + TruncatedSVD dimensionality reduction 
    Preprocessing:
    Scaling
    SVD: The SVD analysis in milestone_2 found ~100 components explain ≥90% of
    variance.
    Models:
    SGDClassifier (log-loss → logistic regression in SGD form)
    SVC(probability=True)
    BaggingClassifier of LinearSVC (wrapped for predict_proba)

    Once again class weights added
    """
    from sklearn.calibration import CalibratedClassifierCV

    sgd = SGDClassifier(
        loss="log_loss",
        max_iter=4000,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced',
        early_stopping=True,
    )

    svc = SVC(kernel="rbf", probability=True, random_state=42, class_weight='balanced')

    bagging = CalibratedClassifierCV(
        BaggingClassifier(
            estimator=LinearSVC(dual=False, max_iter=4000, class_weight='balanced'),
            n_estimators=50,
            max_samples=0.2,
            bootstrap=True,
            bootstrap_features=True,
            n_jobs=-1,
        ),
        cv=3,
    )

    voter = VotingClassifier(
        estimators=[
            ("sgd",     sgd),
            ("svc",     svc),
            ("bagging", bagging),
        ],
        verbose=True,
        n_jobs=-1,
        voting="soft",
    )

    return Pipeline([
        ("flat",  FunctionTransformer(flatten_data, validate=False)),
        ("scale", MinMaxScaler()),
        ("svd",   TruncatedSVD(n_components=n_svd_components, random_state=42)),
        ("model", voter),
    ])


#Wide & Deep Network Preliminary Design
def create_wide_and_deep_pca(meta, n_pca=50):
    '''Wide an Deep architecture
    
    Architecture:
    Input -> Scale
    Wide Path: PCA
    Deep Path: Dense(300, relu) -> Dense(300, relu)
    Softmax
    '''

    n_features = meta["X_shape_"][1]

    inputs = tf.keras.layers.Input(shape=(n_features,))
    wide_input = tf.keras.layers.Lambda(lambda x: x[:, :n_pca])(inputs)
    deep_input = tf.keras.layers.Lambda(lambda x: x[:, n_pca:])(inputs)

    deep = tf.keras.layers.Dense(300, activation="relu")(deep_input)
    deep = tf.keras.layers.Dense(300, activation="relu")(deep)

    concat = tf.keras.layers.Concatenate()([wide_input, deep])
    output = tf.keras.layers.Dense(4, activation="softmax")(concat)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return model

from sklearn.decomposition import PCA
from scikeras.wrappers import KerasClassifier

def build_wide_deep_pipeline(n_components=50):
    """Returns a fresh pipeline so each CV fold starts from scratch."""

    wide_deep_features = FeatureUnion([
        ("pca",         PCA(n_components=n_components, random_state=42)),
        ("passthrough", FunctionTransformer(lambda x: x, validate=False)),
    ])

    clf_wide_pca = KerasClassifier(
        model=create_wide_and_deep_pca,
        model__n_pca=n_components,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping_cb],
        verbose=0,
    )

    return Pipeline([
        ("flat",     flattener),
        ("scale",    MinMaxScaler()),
        ("features", wide_deep_features),
        ("model",    clf_wide_pca),
    ])


#CNN Preliminary Design
def build_cnn():
    """
    Traditional CNN Architecture

    Architecture:
        Conv2D(64, 3, relu) -> MaxPool(2) -> Conv2D(32, 3, relu) ->
        MaxPool(2) -> Flatten -> Dense(32, relu) -> Dense(4, softmax)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def preprocess_for_cnn(images):
    return prep_for_cnn(median(images))


#Resnet Preliminary Design
_DefaultConv2D = partial(
    tf.keras.layers.Conv2D,
    kernel_size=3,
    strides=1,
    padding="same",
    kernel_initializer="he_normal",
    use_bias=False,
)

class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides= 1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            _DefaultConv2D(filters, strides=strides),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            _DefaultConv2D(filters),
            tf.keras.layers.BatchNormalization(),
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                _DefaultConv2D(filters, kernel_size=1, strides=strides),
                tf.keras.layers.BatchNormalization(),
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)


def build_resnet(input_shape=(28, 28, 1), n_classes= 4,learning_rate=1e-3,
):
    """
    Resnet CNN Architecture

    Architecture: 
    Block schedule: Conv(32) -> ResUnit×2(32) -> ResUnit×2(64) -> ResUnit×2(128)
    → GlobalAvgPool -> Flatten -> Dense(n_classes, softmax).
    """
    model = tf.keras.Sequential([
        _DefaultConv2D(32, kernel_size=3, strides=1, input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
    ])

    prev_filters = 32
    for filters in [32, 32, 64, 64, 128, 128]:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=strides))
        prev_filters = filters

    model.add(tf.keras.layers.GlobalAvgPool2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return model

'''
!!!!!!!!

HYPERBAND TUNING

!!!!!!!
'''

#I included the hyperband tuners in here as to keep the final notebook more readable.
#The docstrings at the beginning of each function detail the search space used.
#rationale behind each tuning decision will be detailed in the final report
#finally batch_size=128 was chosen as a universal decision to increase training time.
#This will surely have consequences for generalization however I believe given the timeframe it's worth it

def build_tuned_wide_deep(hp, n_by_threshold, n_max, mean, variance, n_train_samples=22000, batch_size=128, epochs=100):
    """
    Search space:
    pca_threshold 0.90,0.95,0.99,0.999
    n_deep_layers 1-5 dense layers
    deep_units_{i} 64-512 per layer (step 32)
    activation (relu, elu)
    use_batch_norm
    dropout_rate (0.0-0.5) (after each deep dense layer)
    learning_rate log-uniform 1e-4 to 1e-2 (CosineDecay schedule)
    """
    threshold_str  = hp.Choice("pca_threshold", values=["0.90", "0.95", "0.99", "0.999"])
    n_wide         = n_by_threshold[threshold_str]
    activation     = hp.Choice("activation", values=["relu", "elu"])
    n_deep_layers  = hp.Int("n_deep_layers", min_value=1, max_value=3)
    use_batch_norm = hp.Boolean("use_batch_norm")
    dropout_rate   = hp.Float("dropout_rate", min_value=0.0, max_value=0.3, step=0.1)
    learning_rate  = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

    steps_per_epoch = n_train_samples // batch_size
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=steps_per_epoch * epochs,
        warmup_target=learning_rate,
        warmup_steps=steps_per_epoch * 2,
    )

    inputs = tf.keras.layers.Input(shape=(n_max + 784,))

    # Wide path: PCA features up to the chosen variance threshold
    wide = inputs[:, :n_wide]

    # Deep path: raw pixels with internal normalization
    deep = tf.keras.layers.Normalization(mean=mean, variance=variance)(inputs[:, n_max:])
    for i in range(n_deep_layers):
        units = hp.Int(f"deep_units_{i}", min_value=128, max_value=512, step=64)
        deep = tf.keras.layers.Dense(units, activation=activation)(deep)
        if use_batch_norm:
            deep = tf.keras.layers.BatchNormalization()(deep)
        if dropout_rate > 0.0:
            deep = tf.keras.layers.Dropout(dropout_rate)(deep)

    concat = tf.keras.layers.Concatenate()([wide, deep])
    outputs = tf.keras.layers.Dense(4, activation="softmax")(concat)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def build_tuned_cnn(hp, n_train_samples=22000, batch_size=128, epochs=100):
    """
    Search Space for CNNs
    Covers:
    Activation Functions (relu, elu, leaky_relu),
    Number of 'Conv Blocks' 1-3:
        A conv block is defined as a convolutional layer and a pooling layer at a bare minimum
    kernel_size (3 or 5)
    inclusion of batch norm in conv blocks
    dropout_rate and inclusion of dropout rate in conv block (0-.6)
    inclusion and number of dense layers in network (0-2)
    and learning_rate (log sampled from 1e-4-1e-2)
    """
    activation     = hp.Choice("activation", values=["relu", "elu", "leaky_relu"])
    n_conv_blocks  = hp.Int("n_conv_blocks", min_value=2, max_value=3)
    kernel_size    = hp.Choice("kernel_size", values=[3, 5])
    use_batch_norm = hp.Boolean("use_batch_norm")
    dropout_rate   = hp.Float("dropout_rate", min_value=0.0, max_value=0.3, step=0.1)
    n_dense_layers = hp.Int("n_dense_layers", min_value=0, max_value=1)
    learning_rate  = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

    steps_per_epoch = n_train_samples // batch_size
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=steps_per_epoch * epochs,
        warmup_target=learning_rate,
        warmup_steps=steps_per_epoch * 2,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = inputs

    for i in range(n_conv_blocks):
        filters = hp.Choice(f"filters_{i}", values=[32, 64, 128])
        x = tf.keras.layers.Conv2D(
            filters, kernel_size=kernel_size, padding="same", activation=activation
        )(x)
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Flatten()(x)

    for i in range(n_dense_layers):
        dense_units = hp.Choice(f"dense_units_{i}", values=[32, 64, 128])
        x = tf.keras.layers.Dense(dense_units, activation=activation)(x)
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_tuned_resnet(hp, n_train_samples=22000, batch_size=128, epochs=100):
    """
    Search Space for Resnet Architecture

    Covers:
    Activation Functions (relu, elu)
    n_stages (2-4 residual stages)
    base_filters (16 , 32 , 64) 
    units_per_stage  (1-3 ResidualUnits per stage)
    dropout_rate (0.0-0.4 after each stage)
    use_dense: add Dense layer before output
    dense_units (64-256) 
    learning_rate (log-uniform 1e-4 to 1e-2)
    """
    activation      = hp.Choice("activation", values=["relu", "elu", "leaky_relu"])
    n_stages        = hp.Int("n_stages", min_value=2, max_value=3)
    base_filters    = hp.Choice("base_filters", values=[32, 64])
    units_per_stage = hp.Int("units_per_stage", min_value=1, max_value=2)
    dropout_rate    = hp.Float("dropout_rate", min_value=0.0, max_value=0.2, step=0.1)
    use_dense  = hp.Boolean("use_dense")
    learning_rate   = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")


    steps_per_epoch = n_train_samples // batch_size
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=steps_per_epoch * epochs,
        warmup_target=learning_rate,
        warmup_steps=steps_per_epoch * 2,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = inputs

    x = tf.keras.layers.Conv2D(
        base_filters, kernel_size=3, strides=1, padding="same",
        kernel_initializer="he_normal", use_bias=False,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)

    for stage in range(n_stages):
        filters = base_filters * (2 ** stage)
        for unit in range(units_per_stage):
            strides = 2 if (unit == 0 and stage > 0) else 1
            x = ResidualUnit(filters, strides=strides, activation=activation)(x)
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.GlobalAvgPool2D()(x)

    if use_dense:
        dense_head_units = hp.Int("dense_units", min_value=64, max_value=256, step=64)
        x = tf.keras.layers.Dense(dense_head_units, activation=activation)(x)
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model