import matplotlib.pyplot as plt


def plot_trend_and_fft(data, fft_mag, n_plots=5, start_int=101):
    data = data.cpu()
    fft_mag = fft_mag.cpu()
    figure, axis = plt.subplots(n_plots, 2)

    for plot, inter in enumerate(range(start_int, start_int + n_plots)):
        axis[plot, 0].plot(data[:, inter])
        axis[plot, 0].set_title(f'Int. {inter} - Data')

        axis[plot, 1].plot(fft_mag[:, inter])
        axis[plot, 1].set_title(f'Int. {inter} - FFT')

    figure.tight_layout()
