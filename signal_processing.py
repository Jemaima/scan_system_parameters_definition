import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# ========================================================
# Script attributes
# ========================================================

SAVE_LOGS = False

# ========================================================
# system paremeters
# ========================================================
deltaT_counter = 20 * 10 ** (-9)  # s
w_rad = 120 * np.pi

detector_size = np.float64([2.74, 2.74])  # mm
detector_t = 100 * 10 ** (-9)  # s
H_detector = lambda t: 1 / detector_t * np.exp(-t / detector_t)
# ========================================================


def get_detector_params(l, r):
    """

    :param l: distance
    :param r: detector orientation
    :return:
        x_ticks - np.array [mcs]
        ray_t_pass - float, mcs
    """
    axis = 0
    # detector angle size
    current_detector_angle_size = np.arctan(detector_size[axis] * np.cos(r) / l)
    # convert to time dimension
    detector_t_pass = current_detector_angle_size / w_rad

    # number of system counts to register ray scan
    n_counts = (detector_t_pass / deltaT_counter).astype(np.int)

    # number of system tics per scan
    x_ticks = np.linspace(0, detector_t_pass, n_counts) * 10 ** 6

    detector_form = np.array(n_counts * [1])
    return x_ticks, detector_form, detector_t_pass * 10 ** 6


def get_ray_params(l, ray_width, F_max):
    """
    :param l: distance, m
    :param ray_width: width of scanning ray, mm
    :param F_max: peak value
    :return:
    x_ticks - np.array [mcs]
    ray_t_pass - float, mcs

    """
    # ray angle size
    ray_angle_width = np.arctan(ray_width / l)
    # convert to time dimension
    ray_t_pass = ray_angle_width / w_rad
    # number of system counts to register ray scan
    n_counts = (ray_t_pass / deltaT_counter).astype(np.int)
    # number of system tics per scan
    x_ticks = np.linspace(0, ray_t_pass, n_counts) * 10 ** 6
    ray_form = np.array(n_counts * [F_max])
    return x_ticks, ray_form, ray_t_pass * 10 ** 6


def plot_graphics(L, R, ray_width, F_max):
    print("Plotting signal forms")
    try:
        _ = len(ray_width)
        ray_width = ray_width[0]
    except:
        pass
    # detector sizes
    i = 0
    plt.figure()
    for l in L:
        for r in R:
            i = i + 1
            plt.subplot(2, 2, i)
            ticks, d_form, d_width = get_detector_params(l, r)
            plt.plot(ticks, d_form)
            plt.xlabel('mcs')
            plt.title(' '.join(['l =', "{:2.1f}".format(l / 1000), 'm;\nr =', "{:2.1f}".format(r), 'degree\nwidth =',
                                "{:2.2f}".format(d_width), ' mcs']), fontsize=10)
    plt.tight_layout()

    # ray sizes
    i = 0
    plt.figure()
    for l in L:
        i = i + 1
        plt.subplot(1, 2, i)
        r_ticks, r_form, r_width = get_ray_params(l, r, F_max)
        plt.plot(r_ticks, r_form)
        plt.xlabel('mcs')
        plt.title(' '.join(['l =', "{:2.1f}".format(l / 1000), 'm;\nnwidth =', "{:2.2f}".format(r_width),
                            ' mcs']), fontsize=10)
    plt.tight_layout()
    # plt.show()

    # signal sizes
    max_s = int(80 * 10 ** (-6) / deltaT_counter)
    i = 0
    plt.figure()
    for l in L:
        r_ticks, r_form, r_width = get_ray_params(l, ray_width, F_max)
        r_form = r_form * np.random.random(len(r_form)) * noise[0]
        for r in R:
            i = i + 1
            plt.subplot(2, 2, i)
            d_ticks, d_form, d_width = get_detector_params(l, r)
            signal = np.convolve(r_form / sum(r_form), d_form)
            signal = np.pad(signal, (0, int((max_s - len(signal)))),
                            'constant', constant_values=0)
            x_ticks = np.linspace(0, max_s * deltaT_counter * 10 ** 6,
                                  max_s)
            t_real = x_ticks[int(len(d_ticks) / 2 + len(r_ticks) / 2)]
            t = sum(x_ticks * signal) / sum(signal)
            plt.plot(x_ticks, signal)
            plt.xlabel('mcs')
            plt.title(' '.join(
                ['l =', "{:2.1f}".format(l / 1000), 'm;\nr =', "{:2.1f}".format(r * 180 / np.pi), 'degree\ncenter =',
                 "{:2.2f}".format((d_width + r_width)/2), ' mcs']), fontsize=10)

    plt.tight_layout()
    # plt.show()

    # PLOT WITH DETECTOR MPF
    # plt.figure()
    # h_detector_to_convolve = H_detector(np.linspace(0, detector_t, int(detector_t / deltaT_counter)))
    # h_detector_to_convolve = h_detector_to_convolve / sum(h_detector_to_convolve)
    #
    # signal_U = np.convolve(signal, h_detector_to_convolve)
    # x_ticks2 = np.linspace(0, (len(signal) + len(h_detector_to_convolve)) * deltaT_counter * 10 ** 6,
    #                        len(signal) + len(h_detector_to_convolve))[:-1]
    # plt.plot(x_ticks2, signal_U / signal_U.max())
    # plt.xlabel('mcs')
    # plt.ylabel('U(t) normalized')
    # plt.show()


def plot_signal(l,r,ray_width,noise,F_max):
    max_s = int(80 * 10 ** (-6) / deltaT_counter)
    plt.figure()
    r_ticks, r_form, r_width = get_ray_params(l, ray_width, F_max)
    # r_form = r_form * np.random.random(len(r_form)) * noise
    d_ticks, d_form, d_width = get_detector_params(l, r)
    signal = np.convolve(r_form / sum(r_form), d_form)
    signal = np.pad(signal, (0, int((max_s - len(signal)))),
                    'constant', constant_values=0)
    signal = signal+np.random.random(len(signal)) * noise
    x_ticks = np.linspace(0, max_s * deltaT_counter * 10 ** 6,
                          max_s)
    # t_real = x_ticks[int(len(d_ticks) / 2 + len(r_ticks) / 2)]
    # t = sum(x_ticks * signal) / sum(signal)
    plt.plot(x_ticks, signal)
    plt.xlabel('mcs')
    plt.title('l = {:2.1f} m;\nr = {:2.1f} degree\nr_width = {:2.1f} mm\ncenter = {:2.2f} mcs'.format(
        l / 1000, r * 180 / np.pi, ray_width, (d_width + r_width)/2), fontsize=10)
    plt.tight_layout()


def signal_energetic_center(signal, d_ticks, r_ticks, x_ticks):
    t_real = x_ticks[int(len(d_ticks) / 2 + len(r_ticks) / 2)]
    t = sum(x_ticks * signal) / sum(signal)

    return t, t_real


def ray_width_dependence(width_array, L, R, noise, F_max, signal_len = 250):
    print('Getting dataframe for different meanings of ray_width and object position')
    results = pd.DataFrame(columns=['width_array', 'noise', 'distance', 'orientation',
                                    't_real', 't', 't_error'])
    max_s = int(signal_len * 10 ** (-6) / deltaT_counter)
    for width in width_array:

        # signal sizes
        for n in noise:
            for l in L:
                r_ticks, r_form, r_width = get_ray_params(l, width, F_max)
                r_form = r_form
                for r in R:
                    d_ticks, d_form, d_width = get_detector_params(l, r)
                    signal = np.convolve(r_form / sum(r_form), d_form)
                    signal = np.pad(signal, (0, int((max_s - len(signal)))),
                                    'constant', constant_values=0)
                    signal = signal + np.random.random(len(signal)) * n
                    x_ticks = np.linspace(0, max_s * deltaT_counter * 10 ** 6,
                                          max_s)
                    treshold_signal = signal
                    treshold_signal[treshold_signal < n] = 0
                    t_real = x_ticks[int(len(d_ticks) / 2 + len(r_ticks) / 2)]
                    t = sum(x_ticks * treshold_signal) / sum(treshold_signal)
                    results = results.append({'width': width,
                                    'noise': n,
                                    'distance': l,
                                    'orientation': r,
                                    't_real': t_real,
                                    't': t,
                                    't_error': abs(t - t_real)},
                                   ignore_index=True)
        print(' '.join(["width =","{:2.1f}".format(width),'mm complete']))
    if SAVE_LOGS:
        results.to_csv('signal_logs\\results_' + (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '.csv',
                   decimal=',', sep=';')
    return results


if __name__ == "__main__":
    noise = np.array([0.1, 0.2,0.3,0.4, 0.5, 0.6, 0.7])
    ray_width = np.array([10, 20, 30, 40, 50, 60, 70])  # mm
    F_max = 10
    # distance to object
    L = np.array([1, 5]) * 10 ** 3  # mm
    # detector orientation
    R = np.array([0, 60]) * np.pi / 180  # rad

    plot_signal(L[1], R[0], ray_width[1], 0.01, F_max)
    plt.show()
    # plot_signal(L[-1], R[0], ray_width[-1], noise[-1], F_max)
    # plt.show()

    # plot_graphics(L, R, ray_width, F_max)
    # plt.show()

    L = np.array([0.5, 1, 2, 3, 5]) * 10 ** 3  # mm
    # detector orientation
    R = np.array([0, 15, 30, 45, 60]) * np.pi / 180  # rad

    _,_,s_len = get_ray_params(L.min(),ray_width.max(), F_max)

    results = ray_width_dependence(ray_width, L, R, noise, F_max, int(np.ceil(s_len)*2))

