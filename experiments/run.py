import argparse
import matplotlib.pyplot as plt
import experiment
from experiment import Experiment

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
    parser.add_argument("name")
    parser.add_argument("func", choices=["onerun", "graphic"])
    parser.add_argument("config_path")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    try:
        exp = getattr(experiment, args.name.capitalize())(args.config_path)
    except Exception as e:
        raise e

    if args.func == "onerun":
        onerun(exp, args.save)
    elif args.func == "graphic":
        graphic(exp, args.save)
    else:
        raise ValueError("Unknown function {}".format(args.func))
