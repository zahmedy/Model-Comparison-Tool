import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, class_name, title="Confusion Matrix"):
    """
    Return a matplotlib Figure with confusion matrix heatmap.
    """
    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_name)),
        yticks=np.arange(len(class_name)),
        xticklabels=class_name,
        yticklabels=class_name,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )

    # Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add numbers on each cell
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in  range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    
    fig.tight_layout()
    return fig

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def plot_roc_curve(y_true, y_scores, title="ROC Curve"):
    fig, ax = plt.subplots()

    fpr, tpr, _ = roc_curve(y_true, y_scores)  # y_true must be 0/1
    roc_auc = auc(fpr, tpr)

    # Plot line plus markers so points are visible
    ax.plot(fpr, tpr, lw=2, marker="o", markersize=4, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    return fig


def plot_precision_recall(y_true, y_scores, title="Precision-Recall Curve"):
    fig, ax = plt.subplots()

    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    # Plot line plus markers so points are visible
    ax.plot(recall, precision, lw=2, marker="o", markersize=4)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)

    return fig
