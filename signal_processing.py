import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
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
    return x_ticks, detector_form, detector_t_pass


def get_ray_params(l, ray_width, F_max):
    # ray angle size
    ray_angle_width = np.arctan(ray_width / l)
    # convert to time dimension
    ray_t_pass = ray_angle_width / w_rad
    # number of system counts to register ray scan
    n_counts = (ray_t_pass / deltaT_counter).astype(np.int)
    # number of system tics per scan
    x_ticks = np.linspace(0, ray_t_pass, n_counts) * 10 ** 6
    ray_form = np.array(n_counts * [F_max])
    return x_ticks, ray_form, ray_t_pass


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
                                "{:2.2f}".format(d_width * 10 ** 6), ' mcs']), fontsize=10)
    plt.tight_layout()

    # ray sizes
    i = 0
    plt.figure()
    for l in L:
        i = i + 1
        plt.subplot(1, 2, i)
        r_ticks, r_form, r_width = get_detector_params(l, r)
        plt.plot(r_ticks, r_form)
        plt.xlabel('mcs')
        plt.title(' '.join(['l =', "{:2.1f}".format(l / 1000), 'm;\nnwidth =', "{:2.2f}".format(d_width * 10 ** 6),
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
                 "{:2.2f}".format((d_width + r_width) * 10 ** 6), ' mcs']), fontsize=10)

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


def signal_energetic_center(signal, d_ticks, r_ticks, x_ticks):
    t_real = x_ticks[int(len(d_ticks) / 2 + len(r_ticks) / 2)]
    t = sum(x_ticks * signal) / sum(signal)

    return t, t_real


def ray_width_dependence(width_array, L, R, noise, F_max):
    print('Getting dataframe for different meanings of ray_width and object position')
    results = pd.DataFrame(columns=['width_array', 'noise', 'distance', 'orientation',
                                    't_real', 't', 't_error'])
    max_s = int(150 * 10 ** (-6) / deltaT_counter)
    for width in width_array:

        # signal sizes
        for n in noise:
            for l in L:
                r_ticks, r_form, r_width = get_ray_params(l,width,F_max)
                r_form = r_form * np.random.random(len(r_form)) * n
                for r in R:
                    d_ticks, d_form, d_width = get_detector_params(l, r)
                    signal = np.convolve(r_form / sum(r_form), d_form)
                    signal = np.pad(signal, (0, int((max_s - len(signal)))),
                                    'constant', constant_values=0)
                    x_ticks = np.linspace(0, max_s * deltaT_counter * 10 ** 6,
                                          max_s)
                    t_real = x_ticks[int(len(d_ticks) / 2 + len(r_ticks) / 2)]
                    t = sum(x_ticks * signal) / sum(signal)
                    results = results.append({'width_array': width,
                                    'noise': n,
                                    'distance': l,
                                    'orientation': r,
                                    't_real': t_real,
                                    't': t,
                                    't_error': abs(t - t_real)},
                                   ignore_index=True)
        print(' '.join(["width =","{:2.1f}".format(width),'mm complete']))
    results.to_csv('signal_logs\\results_' + (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '.csv',
                   decimal=',', sep=';')
    return results


if __name__ == "__main__":
    noise = [0.001, 0.01, 0.05]
    ray_width = [5,10,20,30]  # mm
    F_max = 10
    # distance to object
    L = np.array([1, 5]) * 10 ** 3  # mm
    # detector orientation
    R = np.array([0, 60]) * np.pi / 180  # rad

    plot_graphics(L, R, ray_width, F_max)
    plt.show()

    results = ray_width_dependence(ray_width, L, R, noise, F_max)

