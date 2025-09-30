import os
import re
import ast
import csv

def parse_metrics_file(filepath):
    """
    Extracts metric dictionaries from a metrics.txt file.
    Returns a dict like { 'NDCG': {...}, 'Recall': {...}, 'Precision': {...}, ... }
    """
    metrics = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            # Match lines like: NDCG: {'NDCG@1': 0.28, ...}
            match = re.match(r"^(\w+):\s*(\{.*\})", line.strip())
            if match:
                metric_name, dict_str = match.groups()
                try:
                    metrics[metric_name] = ast.literal_eval(dict_str)
                except Exception as e:
                    print(f"Skipping malformed line in {filepath}: {line.strip()} ({e})")

    return metrics


def sort_metric_keys(keys):
    """
    Sort metric keys by the numeric part after '@' (e.g., NDCG@1, NDCG@10, ...).
    Keys without '@' are placed at the end.
    """
    def key_func(k):
        if "@" in k:
            try:
                return (0, int(k.split("@")[-1]))
            except ValueError:
                return (0, float("inf"))
        else:
            return (1, k)  # Non-ranked metrics (e.g., MRR) go last, alphabetically
    return sorted(keys, key=key_func)


def write_metrics_to_csv(all_metrics, output_dir):
    """
    Writes separate CSV files for each metric type (NDCG, Recall, etc.)
    all_metrics is {exp_name: {metric: {key: val}, "task": str, "model": str}}
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all metric names across experiments
    metric_names = set()
    for exp_data in all_metrics.values():
        metric_names.update([m for m in exp_data.keys() if m not in ("task", "model")])

    for metric_name in metric_names:
        outpath = os.path.join(output_dir, f"{metric_name}.csv")
        with open(outpath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Collect all keys (like NDCG@1, NDCG@10, etc.)
            all_keys = set()
            for exp_data in all_metrics.values():
                if metric_name in exp_data:
                    all_keys.update(exp_data[metric_name].keys())
            all_keys = sort_metric_keys(all_keys)

            # Header
            writer.writerow(["Task", "Model"] + all_keys)

            # Sort experiments by (task, model)
            sorted_experiments = sorted(
                all_metrics.items(),
                key=lambda kv: (kv[1]["task"], kv[1]["model"])
            )

            # Rows
            for exp_name, exp_data in sorted_experiments:
                if metric_name in exp_data:
                    row = [exp_data["task"], exp_data["model"]] + [
                        exp_data[metric_name].get(k, "") for k in all_keys
                    ]
                    writer.writerow(row)


def main(input_dir, output_dir="csv_metrics"):
    all_metrics = {}

    for filename in os.listdir(input_dir):
        if filename.endswith("_metrics.txt"):
            exp_name = filename.replace("_metrics.txt", "")
            filepath = os.path.join(input_dir, filename)

            # Split experiment name into task and model
            if "-" in exp_name:
                task, model = exp_name.split("-", 1)
            else:
                task, model = exp_name, "unknown"

            metrics = parse_metrics_file(filepath)
            if metrics:
                metrics["task"] = task
                metrics["model"] = model
                all_metrics[exp_name] = metrics

    write_metrics_to_csv(all_metrics, output_dir)
    print(f"Metrics written to {output_dir}/ (one CSV per metric type)")



if __name__ == "__main__":
    # Example usage: adjust input_dir to where your collected metrics are
    main("/u/poellhul/Documents/Masters/VAULT/results", "/u/poellhul/Documents/Masters/VAULT/results")
