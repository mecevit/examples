"""Titanic Survival Project Example

This file demonstrates how we can develop and train our model by using the
`passenger_features` we've developed earlier. Every ML model project
should have a definition file like this one.

"""
from typing import Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from layer import Featureset, Train


def train_model(train: Train, pf: Featureset("passenger_features")) -> Any:
    """Model train function

    This function is a reserved function and will be called by Layer
    when we want this model to be trained along with the parameters.
    Just like the `passenger_features` featureset, you can add more
    parameters to this method to request artifacts (datasets,
    featuresets or models) from Layer.

    Args:
        train (layer.Train): Represents the current train of the model, passed by
            Layer when the training of the model starts.
        pf (layer.Featureset): Layer will return a Featureset object,
            an interface to access the features inside the
            `passenger_features`

    Returns:
       model: Trained model object

    """
    df = pf.to_pandas()

    X = df.drop(["PassengerId", "Survived"], axis=1)
    y = df["Survived"]

    # Split dataset into training set and test set
    test_size = 0.2
    train.log_parameter("test_size", test_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size)

    # Here we register input & output of the train. Layer will use
    # this registers to extract the signature of the model and calculate
    # the drift
    train.register_input(X_train)
    train.register_output(y_train)

    # We can easily do hyperparameter tuning to find the best parameters for the
    # best accuracy. See the `hyperparameters` section in the `model.yml` file
    # in this folder.
    estimators = train.get_parameter("n_estimators")
    max_depth = train.get_parameter("max_depth")
    max_samples = train.get_parameter("max_samples")

    # We will use `RandomForestClassifier` for this task
    random_forest = RandomForestClassifier(
        n_estimators=estimators,
        max_depth=max_depth,
        max_samples=max_samples
    )

    # We fit our model with the train and the label data
    random_forest.fit(X_train, y_train)

    # Let's calculate the accuracy of our model
    y_pred = random_forest.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # We can log metrics as well to be compared later in Layer Web interface
    # `accuracy` metric will also be our objective for hyper parameter tuning
    # See the `hyperparameters` section in the `model.yml` file in this folder
    train.log_metric("accuracy", acc)

    # We return the model
    return random_forest
