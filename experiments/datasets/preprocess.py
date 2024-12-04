import os
import pathlib
import pickle
import sys

from el0ps.utils import compute_lmbd_max

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instances import calibrate_parameters, process_data  # noqa


def get_dataset_path(dataset):
    dataset_name = dataset["dataset_opts"]["dataset_name"]
    dataset_dir = pathlib.Path(__file__).parent.absolute()
    dataset_path = dataset_dir.joinpath(dataset_name).with_suffix(".pkl")
    return dataset_path


def load_dataset(dataset):
    dataset_path = get_dataset_path(dataset)
    with open(dataset_path, "rb") as dataset_file:
        data = pickle.load(dataset_file)
        A = data["A"]
        y = data["y"]
        if "x_true" in data.keys():
            x_true = data["x_true"]
        else:
            x_true = None
        if "calibrations" in data.keys():
            calibrations = data["calibrations"]
        else:
            calibrations = []
    return A, y, x_true, calibrations


def save_dataset(dataset, A, y, x_true, calibrations):
    dataset_path = get_dataset_path(dataset)
    data = {"A": A, "y": y, "x_true": x_true, "calibrations": calibrations}
    with open(dataset_path, "wb") as dataset_file:
        pickle.dump(data, dataset_file)


def preprocess_dataset(dataset):
    A, y, x_true, calibrations = load_dataset(dataset)
    print("Preprocessing dataset...")
    A, y, x_true = process_data(
        dataset["datafit_name"],
        A,
        y,
        x_true,
        dataset["process_opts"]["center"],
        dataset["process_opts"]["normalize"],
    )
    save_dataset(dataset, A, y, x_true, calibrations)


def calibrate_dataset(dataset):

    A, y, x_true, calibrations = load_dataset(dataset)

    for calibration in calibrations:
        if (
            str(calibration["datafit"]) == dataset["datafit_name"] and 
            str(calibration["penalty"]) == dataset["penalty_name"]
        ):
            print("Calibration found")
            return

    print("Calibrating dataset...")
    datafit, penalty, lmbd, x_cal = calibrate_parameters(
        dataset["datafit_name"],
        dataset["penalty_name"],
        A,
        y,
        x_true,
    )
    calibrations.append(
        {
            "datafit_name": dataset["datafit_name"],
            "penalty_name": dataset["penalty_name"],
            "datafit": datafit,
            "penalty": penalty,
            "lmbd": lmbd,
            "x_cal": x_cal,
        }
    )

    save_dataset(dataset, A, y, x_true, calibrations)


def reset_calibrations(dataset, all=False):
    print("Resetting calibrations...")
    A, y, x_true, calibrations = load_dataset(dataset)
    if all:
        calibrations = []
    else:
        calibrations = [
            c for c in calibrations if (
                str(c["datafit"]) != dataset["datafit_name"] or
                str(c["penalty"]) != dataset["penalty_name"]
            )
        ]
    save_dataset(dataset, A, y, x_true, calibrations)


def display_calibrations(dataset):
    A, _, _, calibrations = load_dataset(dataset)
    print("Dataset:", dataset["dataset_opts"]["dataset_name"])
    if len(calibrations) == 0:
        print("No calibrations found.")
        return
    for calibration in calibrations:
        datafit = calibration["datafit"]
        penalty = calibration["penalty"]
        lmbd = calibration["lmbd"]
        x_cal = calibration["x_cal"]
        lmbd_max = compute_lmbd_max(datafit, penalty, A)
        print(
            "  {} / {}".format(
                str(calibration["datafit"]),
                str(calibration["penalty"]),
            )
        )
        print("    lratio: {:.2e}".format(lmbd / lmbd_max))
        print("    num nz: {}".format(sum(x_cal != 0.0)))
        print("    params:")
        for param_name, param_value in penalty.params_to_dict().items():
            print("      {:<6}: {}".format(param_name, param_value))


if __name__ == "__main__":

    dataset = {
        "dataset_type": "hardcoded",
        "dataset_opts": {"dataset_name": "riboflavin"},
        "process_opts": {"center": True, "normalize": True},
        "datafit_name": "Leastsquares",
        "penalty_name": "Bigm",
    }

    preprocess_dataset(dataset)
    reset_calibrations(dataset, all=False)
    # calibrate_dataset(dataset)
    display_calibrations(dataset)
