
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true,
    y_pred,
    y_prob,
    num_classes=8
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),
        "micro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0
            )
        ),
    }

    y_true_one_hot = np.eye(
        num_classes,
        dtype=np.float32
    )[y_true]

    metrics["macro_auc_pr"] = float(
        average_precision_score(
            y_true_one_hot,
            y_prob,
            average="macro"
        )
    )

    return metrics


def compute_confusion_matrix(
    y_true,
    y_pred,
    num_classes=8
):
    labels = list(range(num_classes))

    return confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )


def compute_per_class_f1(
    y_true,
    y_pred,
    num_classes=8
):
    labels = list(range(num_classes))

    return f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0
    )
