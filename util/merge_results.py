import json


def merge_results(paths, aliases):
    """
    Merges the test_result.json files written by main/test.py.

    paths: list of str, the test_result.json of each run
    aliases: list of str, the name of each run, in the same order as paths

    Returns {experiment: {alias: metrics}}, e.g.
    {"zeroshot_classification_dtd": {"baseline": {"acc1": 0.39, ...}, "finetuned": {...}}, ...}.
    An experiment missing from a run is left out of that run's entry.
    """
    assert len(paths) == len(aliases), "paths and aliases must have the same length"
    assert len(set(aliases)) == len(aliases), "aliases must be unique"
    merged = {}
    for path, alias in zip(paths, aliases):
        with open(path, "r") as f:
            results = json.load(f)
        for experiment, metrics in results.items():
            merged.setdefault(experiment, {})[alias] = metrics
    return merged
