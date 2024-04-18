import argparse
import matplotlib.pyplot as plt
import pathlib
import sys

sys.path.append(pathlib.Path(__file__).parent.parent.absolute())
from experiments.experiment import (  # noqa: E402
    Experiment,
    Perfprofile,
    Regpath,
    Statistics,
    RelaxQuality,
)

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)


def onerun(exp: Experiment, save=True):
    exp.setup()
    exp.load_problem()
    exp.calibrate_parameters()
    exp.precompile()
    exp.run()
    if save:
        exp.save_results()


def graphic(exp: Experiment, save=False):
    exp.setup()
    exp.load_results()
    if save:
        exp.save_plot()
    else:
        exp.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name", choices=[
            "relaxquality", 
            "perfprofile", 
            "regpath", 
            "statistics",
        ]
    )
    parser.add_argument("func", choices=["onerun", "graphic"])
    parser.add_argument("config_path")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if args.name == "relaxquality":
        exp = RelaxQuality(args.config_path)
    elif args.name == "perfprofile":
        exp = Perfprofile(args.config_path)
    elif args.name == "regpath":
        exp = Regpath(args.config_path)
    elif args.name == "statistics":
        exp = Statistics(args.config_path)
    else:
        raise ValueError("Unknown experiment {}".format(args.name))

    if args.func == "onerun":
        onerun(exp, args.save)
    elif args.func == "graphic":
        graphic(exp, args.save)
    else:
        raise ValueError("Unknown function {}".format(args.func))
