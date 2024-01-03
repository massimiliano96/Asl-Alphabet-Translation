import os

import matplotlib.pyplot as plt


def plot_metrics(categories, values):
    plt.bar(categories, values)
    plt.title("Metrics")

    # Add values on top of each bar
    for i in range(len(categories)):
        plt.text(categories[i], values[i], str(values[i]), ha="center", va="bottom")

    path = "output/test"
    if os.path.exists(path):
        os.removedirs(path)
    os.mkdir(path)

    # Display the plot
    plt.savefig(str(path + "/metrics_test.png"))
    plt.close()
