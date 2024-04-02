import argparse
import matplotlib.pyplot as plt
import pathlib
import sys

sys.path.append(pathlib.Path(__file__).parent.parent.absolute())
from experiments.experiment import Experiment  # noqa: E402
from experiments.icml.perfprofile import Perfprofile  # noqa: E402
from experiments.icml.realworld import Realworld  # noqa: E402
from experiments.icml.statistics import Statistics  # noqa: E402

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)


def onerun(exp: Experiment, save=True):
    exp.setup()
    exp.load_problem()
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
        "name", choices=["perfprofile", "realworld", "statistics"]
    )
    parser.add_argument("func", choices=["onerun", "graphic"])
    parser.add_argument("config_path")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if args.name == "perfprofile":
        exp = Perfprofile(args.config_path)
    elif args.name == "realworld":
        exp = Realworld(args.config_path)
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
