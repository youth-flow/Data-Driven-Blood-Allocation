"""Use the common morning-data runner while retaining this model namespace."""

from dat_study.runner import run


def run_experiment(config, fold, run_name, smoke=False, recipe_dir=None):
    return run(config, fold, ["Set-ETO"], run_name, smoke=smoke, recipe_dir=recipe_dir)
