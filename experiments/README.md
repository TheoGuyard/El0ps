Experiments
===========

This directory contains experiments published in papers linked to `el0ps`.
An experiment can be run from the root folder of `el0ps` using the command

```bash
$ "python experiments/run.py <experiment_name> onerun <path_to_setup_file>"
```

where `<experiment_name>` is the name of the experiment (`perfprofile`,`regpath` or `statistics`) and ``<path_to_setup_file>`` is the path to the setup file.
The option `--save` allows saving the results of the experiment in the `results` folder.
Graphics can be generated from saved experiments matching a given configuration using the command

```bash
$ "python experiments/run.py <experiment_name> graphic <path_to_setup_file>"
```

and the option `--save` allows saving the results of the experiment in the `saves` folder.
Examples of configuration files can be found in the `experiments/icml` folder.