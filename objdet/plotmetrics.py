import json
import matplotlib.pyplot as plt


def load_json_arr(json_path):
    lines = []
    with open(json_path, "r") as f:
        for line in f:
            line = json.loads(line)
            if "total_loss" in line.keys() or "validation_loss" in line.keys():
                lines.append(line)
    return lines


def display(save_path, fig=None):
    if not save_path:
        plt.show()
    else:
        with open(save_path, "wb") as f:
            if fig:
                fig.savefig(f)
                plt.close(fig)
            else:
                plt.savefig(f)
            f.flush()


def plot_losses(experiment_folder, save_path):
    experiment_metrics = load_json_arr(experiment_folder + "/metrics.json")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(
        [x["iteration"] for x in experiment_metrics if "validation_loss" in x],
        [x["validation_loss"] for x in experiment_metrics if "validation_loss" in x],
    )
    ax.plot(
        [x["iteration"] for x in experiment_metrics],
        [x["total_loss"] for x in experiment_metrics],
    )

    ax.legend(["total_loss", "validation_loss"], loc="upper left")

    display(save_path, fig)
