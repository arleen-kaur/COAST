import math


METRIC_KEYS = ("warm_ndcg", "warm_hr", "cold_ndcg", "cold_hr")


def _mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def aggregate_seed_runs(seed_runs: list[dict]) -> dict:
    methods = {}
    for run in seed_runs:
        for method, metrics in run.items():
            methods.setdefault(method, {k: [] for k in METRIC_KEYS})
            for key in METRIC_KEYS:
                if key in metrics:
                    methods[method][key].append(metrics[key])

    out = {}
    for method, buckets in methods.items():
        out[method] = {}
        for key in METRIC_KEYS:
            if not buckets[key]:
                continue
            mean, std = _mean_std(buckets[key])
            out[method][key] = round(mean, 4)
            out[method][f"{key}_std"] = round(std, 4)
            out[method][f"{key}_per_seed"] = {
                str(i): round(v, 4) for i, v in enumerate(buckets[key])
            }
    return out
