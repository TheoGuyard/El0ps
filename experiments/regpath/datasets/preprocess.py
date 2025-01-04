import pathlib
import pickle

from experiments.instance import preprocess_data, calibrate_parameters


def get_dataset_path(dataset):
    dataset_name = dataset["dataset_name"]
    dataset_dir = pathlib.Path(__file__).parent.absolute()
    dataset_path = dataset_dir.joinpath(dataset_name).with_suffix(".pkl")
    return dataset_path


def load_dataset(dataset):
    dataset_path = get_dataset_path(dataset)
    with open(dataset_path, "rb") as dataset_file:
        data = pickle.load(dataset_file)
        A = data["A"]
        y = data["y"]
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
    A, y, x_true = preprocess_data(
        A,
        y,
        x_true,
        dataset["process_opts"]["center"],
        dataset["process_opts"]["normalize"],
        y_binary=dataset["datafit_name"] in ["Logistic", "Squaredhinge"],
    )
    save_dataset(dataset, A, y, x_true, calibrations)


def calibrate_dataset(dataset):

    A, y, x_true, calibrations = load_dataset(dataset)

    for calibration in calibrations:
        if calibration["dataset"] == dataset:
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
            "dataset": dataset,
            "penalty_params": penalty.params_to_dict(),
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
        calibrations = [c for c in calibrations if c["dataset"] != dataset]
    save_dataset(dataset, A, y, x_true, calibrations)


def display_calibrations(dataset):
    A, _, _, calibrations = load_dataset(dataset)
    print("Dataset:", dataset["dataset_name"])
    if len(calibrations) == 0:
        print("No calibrations found.")
        return
    for calibration in calibrations:
        datafit = calibration["dataset"]["datafit_name"]
        penalty = calibration["dataset"]["penalty_name"]
        lmbd = calibration["lmbd"]
        x_cal = calibration["x_cal"]
        print("  {} / {}".format(datafit, penalty))
        print("    lambda : {:.2e}".format(lmbd))
        print("    num nz : {}".format(sum(x_cal != 0.0)))
        print("    penalty: {}".format(calibration["penalty_params"]))


if __name__ == "__main__":

    dataset = {
        "dataset_name": "bctcga",
        "datafit_name": "Leastsquares",
        "penalty_name": "BigmL2norm",
        "process_opts": {"center": True, "normalize": True},
    }

    preprocess_dataset(dataset)
    reset_calibrations(dataset)
    calibrate_dataset(dataset)
    display_calibrations(dataset)
