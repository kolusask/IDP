import matplotlib.pyplot as plt

from config import *


def plot_trend_and_fft(data, fft_mag, n_plots=5, start_sec=101):
    data = data.cpu()
    fft_mag = fft_mag.cpu()
    figure, axis = plt.subplots(n_plots, 2)

    for plot, sec in enumerate(range(start_sec, start_sec + n_plots)):
        axis[plot, 0].plot(data[:, sec])
        axis[plot, 0].set_title(f'Sec. {sec} - Data')

        axis[plot, 1].plot(fft_mag[:, sec])
        axis[plot, 1].set_title(f'Sec. {sec} - FFT')

    figure.tight_layout()
