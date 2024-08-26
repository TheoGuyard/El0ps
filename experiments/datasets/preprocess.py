import numpy as np
import os
import pathlib
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instances import calibrate_parameters, process_data  # noqa


def preprocess_dataset(
    dataset_path,
    datafit_name,
    penalty_name,
    normalize=True,
    center=True,
):

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

    A, y, x_true = process_data(
        datafit_name,
        penalty_name,
        A,
        y,
        x_true,
        center,
        normalize,
    )

    data = {
        "A": A,
        "y": y,
        "x_true": x_true,
        "calibrations": calibrations,
    }
    with open(dataset_path, "wb") as dataset_file:
        pickle.dump(data, dataset_file)


def calibrate_dataset(
    dataset_path,
    datafit_name,
    penalty_name,
    normalize=True,
    center=True,
):

    with open(dataset_path, "rb") as dataset_file:
        data = pickle.load(dataset_file)
        A = np.copy(data["A"])
        y = np.copy(data["y"])
        if "x_true" in data.keys():
            x_true = np.copy(data["x_true"])
        else:
            x_true = None
        if "calibrations" in data.keys():
            calibrations = data["calibrations"]
        else:
            calibrations = []

    if np.any(
        [
            calibration["datafit_name"] == datafit_name
            and calibration["penalty_name"] == penalty_name
            and calibration["normalize"] == normalize
            and calibration["center"] == center
            for calibration in calibrations
        ]
    ):
        print(
            "Calibration found for {}/{}/{}/{}".format(
                datafit_name, penalty_name, center, normalize
            )
        )
        return

    print(
        "Calibrating {}/{}/{}/{}/{}...".format(
            dataset_name, datafit_name, penalty_name, center, normalize
        )
    )
    _, penalty, lmbd, x_cal = calibrate_parameters(
        datafit_name,
        penalty_name,
        A,
        y,
        x_true,
    )
    print("  num nz: {}".format(sum(x_cal != 0.0)))
    for param_name, param_value in penalty.params_to_dict().items():
        if param_name in ["x_lb", "x_ub"]:
            print(
                "  {}\t: {}".format(
                    param_name, np.linalg.norm(param_value, np.inf)
                )
            )
        else:
            print("  {}\t: {}".format(param_name, param_value))
    calibrations.append(
        {
            "datafit_name": datafit_name,
            "penalty_name": penalty_name,
            "normalize": normalize,
            "center": center,
            "penalty_params": penalty.params_to_dict(),
            "lmbd": lmbd,
            "x_cal": x_cal,
        }
    )

    data = {
        "A": A,
        "y": y,
        "x_true": x_true,
        "calibrations": calibrations,
    }
    with open(dataset_path, "wb") as dataset_file:
        pickle.dump(data, dataset_file)


def reset_calibrations(dataset_path):

    with open(dataset_path, "rb") as dataset_file:
        data = pickle.load(dataset_file)
        A = np.copy(data["A"])
        y = np.copy(data["y"])
        if "x_true" in data.keys():
            x_true = np.copy(data["x_true"])
        else:
            x_true = None
        if "calibrations" in data.keys():
            calibrations = data["calibrations"]
        else:
            calibrations = []

    data = {
        "A": A,
        "y": y,
        "x_true": x_true,
        "calibrations": [],
    }
    with open(dataset_path, "wb") as dataset_file:
        pickle.dump(data, dataset_file)


if __name__ == "__main__":

    dataset_name = "arcene"
    dataset_dir = pathlib.Path(__file__).parent.absolute()
    dataset_path = dataset_dir.joinpath(dataset_name).with_suffix(".pkl")
    datafit_name = "Squaredhinge"
    penalty_name = "BoundsConstraint"

    preprocess_dataset(dataset_path, datafit_name, penalty_name)
    calibrate_dataset(dataset_path, datafit_name, penalty_name)
