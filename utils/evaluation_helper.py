import numpy as np

from .evaluate_12ECG_score import (
    compute_accuracy,
    compute_auc,
    compute_beta_measures,
    compute_challenge_metric,
    compute_f_measure,
    load_weights,
    replace_equivalent_classes,
)


def evaluate_score_batch(
    predicted_classes=[],  # list, len(num_classes), str(code)
    predicted_labels=[],  # shape (num_examples, num_classes), T/F for each code
    predicted_probabilities=[],  # shape (num_examples, num_classes), prob. [0-1] for each class
    raw_ground_truth_labels=[],  # list(('284470004', '5911801'), ('164890007',), ...) length == num_examples
    weights_file="evaluation-2020/weights.csv",
    normal_class="426783006",
    equivalent_classes=[
        ["713427006", "59118001"],
        ["284470004", "63593006"],
        ["427172004", "17338001"],
    ],
):
    """This is a helper function for getting
    auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric
    without needing the directories of labels and prediction outputs.
    It is useful for directly calculating the scores given the
    classes, predicted labels, and predicted probabilities.
    """

    classes, weights = load_weights(weights_file, equivalent_classes)
    labels = _load_labels(
        [tuple(v) for v in raw_ground_truth_labels],
        predicted_classes,
        weight_classes=classes,
        equivalent_classes=equivalent_classes,
    )

    binary_outputs, scalar_outputs = _load_outputs(
        predicted_classes,
        predicted_labels,
        predicted_probabilities,
        weight_classes=classes,
        equivalent_classes=equivalent_classes,
    )

    # import pdb; pdb.set_trace()
    auroc, auprc, auroc_classes, auprc_classes = compute_auc(labels, scalar_outputs)
    accuracy = compute_accuracy(labels, binary_outputs)
    f_measure, f_measure_classes = compute_f_measure(labels, binary_outputs)
    f_beta_measure, g_beta_measure = compute_beta_measures(
        labels, binary_outputs, beta=2
    )
    challenge_metric = compute_challenge_metric(
        weights, labels, binary_outputs, classes, normal_class
    )
    return (
        classes,
        auroc,
        auprc,
        auroc_classes,
        auprc_classes,
        accuracy,
        f_measure,
        f_measure_classes,
        f_beta_measure,
        g_beta_measure,
        challenge_metric,
    )

    # return (
    #     auroc,
    #     auprc,
    #     accuracy,
    #     f_measure,
    #     f_beta_measure,
    #     g_beta_measure,
    #     challenge_metric,
    # )


def _load_labels(
    raw_ground_truth_labels, predicted_classes, weight_classes, equivalent_classes
):
    num_recordings = len(raw_ground_truth_labels)
    num_classes = len(weight_classes)

    # Load diagnoses.
    tmp_labels = list()
    for i in range(num_recordings):
        dxs = [str(v) for v in raw_ground_truth_labels[i]]
        dxs = replace_equivalent_classes(dxs, equivalent_classes)
        tmp_labels.append(dxs)

    # Use one-hot encoding for labels.
    labels = np.zeros((num_recordings, num_classes), dtype=np.bool)
    for i in range(num_recordings):
        dxs = tmp_labels[i]
        for j, x in enumerate(weight_classes):
            if str(x) in dxs:
                labels[i, j] = 1

    return labels


def _load_outputs(
    predicted_classes,
    predicted_labels,
    predicted_probabilities,
    weight_classes,
    equivalent_classes,
):
    num_recordings = len(predicted_probabilities)
    num_classes = len(weight_classes)

    tmp_labels = []
    row = replace_equivalent_classes(
        [str(v) for v in predicted_classes], equivalent_classes
    )
    tmp_labels = [row for _ in range(num_recordings)]
    tmp_binary_outputs = predicted_labels
    tmp_scalar_outputs = predicted_probabilities

    # Use one-hot encoding for binary outputs and the same order for scalar outputs.
    # If equivalent classes have different binary outputs, then the representative class is positive if any equivalent class is positive.
    # If equivalent classes have different scalar outputs, then the representative class is the mean of the equivalent classes.
    binary_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    scalar_outputs = np.zeros((num_recordings, num_classes), dtype=np.float64)

    for i in range(num_recordings):
        dxs = tmp_labels[i]
        for j, x in enumerate(weight_classes):
            indices = [k for k, y in enumerate(dxs) if str(x) == y]
            if indices:
                binary_outputs[i, j] = np.any(
                    [tmp_binary_outputs[i][k] for k in indices]
                )
                scalar_outputs[i, j] = np.nanmean(
                    [tmp_scalar_outputs[i][k] for k in indices]
                )

    # If any of the outputs is a NaN, then replace it with a zero.
    binary_outputs[np.isnan(binary_outputs)] = 0
    scalar_outputs[np.isnan(scalar_outputs)] = 0

    return binary_outputs, scalar_outputs
