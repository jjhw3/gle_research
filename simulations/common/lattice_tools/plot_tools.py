import matplotlib.pyplot as plt


def force_aspect(aspect=1.0):
    plt.gca().set_aspect(aspect)


def cla():
    plt.clf()
    plt.cla()
    plt.close()