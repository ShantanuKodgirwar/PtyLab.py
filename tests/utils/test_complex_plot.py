import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from PtyLab.utils.visualisation import complex2rgb, complexPlot


def test_complex_plot():
    testComplexArray = np.ones((100, 100)) * (1 + 1j)
    testRGBArray = complex2rgb(testComplexArray)
    complexPlot(testRGBArray)
    plt.close("all")
