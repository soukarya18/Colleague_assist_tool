import json
from pathlib import Path


METRICS_FILE = Path("data/metrics.json")


def load_metrics() -> list:

    if not METRICS_FILE.exists():
        return []

    with open(
        METRICS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data.get("metrics", [])


def save_metrics(metrics: list) -> None:

    METRICS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data = {
        "metrics": metrics
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


def show_metrics(metrics: list) -> None:

    if not metrics:

        print("\nNo metrics found.")

        return

    print("\nCurrent Metrics")
    print("------------------------------")

    for index, metric in enumerate(
        metrics,
        start=1
    ):

        print(f"\n{index}. {metric['name']}")

        print(
            f"   Type   : "
            f"{metric['metric_type']}"
        )

        print(
            f"   Weight : "
            f"{metric['weight']}"
        )

        print("   Criteria:")

        for criteria in metric["criteria"]:

            print(
                f"   - {criteria}"
            )


def add_metric(metrics: list) -> None:

    print("\nAdd New Metric")
    print("------------------------------")

    name = input(
        "Metric name: "
    ).strip()

    metric_type = input(
        "Metric type (Fatal/Non-Fatal): "
    ).strip()

    weight = float(
        input("Weight: ")
    )

    criteria = input(
        "Criteria: "
    ).strip()

    metric = {
        "name": name,
        "metric_type": metric_type,
        "criteria": [criteria],
        "weight": weight
    }

    metrics.append(metric)

    print(
        "\nMetric added successfully."
    )


def delete_metric(metrics: list) -> None:

    if not metrics:

        print(
            "\nNo metrics available."
        )

        return

    show_metrics(metrics)

    choice = int(
        input(
            "\nEnter metric number to delete: "
        )
    )

    if choice < 1 or choice > len(metrics):

        print(
            "\nInvalid metric number."
        )

        return

    deleted_metric = metrics.pop(
        choice - 1
    )

    print(
        f"\nDeleted: "
        f"{deleted_metric['name']}"
    )


def update_metric(metrics: list) -> None:

    if not metrics:

        print(
            "\nNo metrics available."
        )

        return

    show_metrics(metrics)

    choice = int(
        input(
            "\nEnter metric number to update: "
        )
    )

    if choice < 1 or choice > len(metrics):

        print(
            "\nInvalid metric number."
        )

        return

    metric = metrics[choice - 1]

    print(
        f"\nUpdating: {metric['name']}"
    )

    print(
        "Press Enter to keep the existing value."
    )

    name = input(
        f"Name [{metric['name']}]: "
    ).strip()

    if name:
        metric["name"] = name

    metric_type = input(
        f"Type [{metric['metric_type']}]: "
    ).strip()

    if metric_type:
        metric["metric_type"] = metric_type

    weight = input(
        f"Weight [{metric['weight']}]: "
    ).strip()

    if weight:
        metric["weight"] = float(weight)

    criteria = input(
        f"Criteria [{metric['criteria'][0]}]: "
    ).strip()

    if criteria:
        metric["criteria"] = [criteria]

    print(
        "\nMetric updated successfully."
    )


def modify_metrics(metrics: list) -> bool:

    changed = False

    while True:

        print("\nMetric Management")
        print("------------------------------")

        print("1. Add metric")
        print("2. Delete metric")
        print("3. Update metric")
        print("4. Done")

        choice = input(
            "\nChoose an option: "
        ).strip()

        if choice == "1":

            add_metric(metrics)

            changed = True

        elif choice == "2":

            delete_metric(metrics)

            changed = True

        elif choice == "3":

            update_metric(metrics)

            changed = True

        elif choice == "4":

            break

        else:

            print(
                "\nInvalid option."
            )

    return changed


def main() -> None:

    metrics = load_metrics()

    show_metrics(metrics)

    choice = input(
        "\nDo you want to modify the metrics? (y/n): "
    ).strip().lower()

    if choice == "y":

        changed = modify_metrics(
            metrics
        )

        if changed:

            save_metrics(
                metrics
            )

            print(
                "\nMetrics saved successfully."
            )

    else:

        print(
            "\nNo changes made."
        )


if __name__ == "__main__":
    main()