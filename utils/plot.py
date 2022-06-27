import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

plt.style.use('seaborn-darkgrid')


def plot_learning_curve(out, epochs, **kwargs):
    epochs = np.arange(epochs) + 1
    labels = list(kwargs.keys())

    xmajorLocator = MultipleLocator(10)
    xmajorFormatter = FormatStrFormatter("%2d")
    xminorLocator   = MultipleLocator(5)
    ymajorLocator = MultipleLocator(0.1)
    ymajorFormatter = FormatStrFormatter('%.1f')
    yminorLocator   = MultipleLocator(0.05)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=120)
    ax.set_title("Accuracy")
    ax.grid(axis="y", linestyle="--")

    handles = []
    for i, (data, label) in enumerate(zip(kwargs.values(), kwargs.keys())):
        handle, = ax.plot(np.arange(len(data))+1, data, label=label, color=f"C{i}")
        handles.append(handle)
        x, y = np.argmax(data) + 1, np.max(data)
        ax.plot(x, y, marker="*", markersize=8, color="red")
        ax.annotate(f"{x}: {y:4.2%}", xy=(x, y), xytext=(x*0.97, y*1.03))
    ax.legend(handles, labels)

    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_major_formatter(ymajorFormatter)
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.set_xlim(0, len(epochs)+1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("ACC")
    ax.set_xlabel("# of Epoch")
    ax.set_xticklabels([])

    plt.savefig(f"{out}/LearningCurve.png", format="png")
    plt.close()