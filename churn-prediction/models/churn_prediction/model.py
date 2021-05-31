"""Churn Prediction Project Example

This file demonstrates how we can develop and train our model by using the
`user_features` we've developed earlier. Every ML model project
should have a definition file like this one.

"""
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import layer
from layer import Featureset, Train


def train_model(
    train: Train,
    uf: Featureset(
        "user_features",
        feature_names=[
            "count_login_sum_7d_every_7d",
            "count_help_view_sum_7d_every_7d",
            "count_thumbsup_sum_14d_every_14d",
            "count_thumbsdown_sum_14d_every_14d",
            "count_error_sum_7d_every_7d",
        ],
    ),
    churned: Featureset(
        "user_features",
        feature_names=[
            "is_churned",
        ],
    ),
) -> Any:
    # Featureset includes all historical values of the every user.
    # For each user, sort it by the aggregation timestamp
    user_events = uf.to_pandas().sort_values(["userId", "timestamp"]).fillna(0)

    # Instead of timestamps, use number of days since the user was active
    user_events = user_events.merge(
        user_events[["userId", "timestamp"]].groupby("userId").min(),
        on="userId",
        how="left",
    )
    user_events["active_days"] = (
        user_events["timestamp_x"] - user_events["timestamp_y"]
    ).dt.days
    user_events.drop(["timestamp_x", "timestamp_y"], axis=1, inplace=True)

    # Get user churn data
    is_churned = churned.to_pandas()

    # Align the data for model train
    data = user_events.merge(is_churned, on="userId", how="left")
    x = data.drop(["userId", "is_churned"], axis=1)
    y = data["is_churned"]

    test_size = 0.2
    train.log_parameter("test_size", test_size)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    # Register input and target data
    train.register_input(x_train)
    train.register_output(y_train)

    # Train our model
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # We log the `accuracy` metric
    score = model.score(x_test, y_test)
    train.log_metric("accuracy", score)

    return model
