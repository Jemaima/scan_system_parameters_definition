import numpy as np
import matplotlib.pyplot as plt

deltaT_counter = 20 * 10 ** (-9)  # s
detector_size = np.float64([2.74, 2.74])  # mm
detector_t = 100 * 10 ** (-9)  # s

H_detector = lambda t: 1 / detector_t * np.exp(-t / detector_t)


w_rad = 120 * np.pi

signal_width = 5  # mm
F_max = 10
# distance to object
L = np.array([10 ** 3, 3 * 10 ** 3, 5 * 10 ** 3])  # mm
# detector orientation
R = np.array([0, 15, 30, 45, 60]) * np.pi / 180  # rad

cur_dist = L[0]
cur_angle = R[4]
axis = 0

# detector angle size
current_detector_angle_size = np.arctan(detector_size[axis] * np.cos(cur_angle) / cur_dist)
# ray angle size
ray_angle_width = np.arctan(signal_width / cur_dist)

# convert to time dimension
ray_t_pass = ray_angle_width / w_rad
detector_t_pass = current_detector_angle_size / w_rad

# number of system counts to register ray scan
n_counts = ((np.array([ray_t_pass, detector_t_pass])) / deltaT_counter).astype(np.int)

# if ray wider then detector
if n_counts[0] > n_counts[1]:
    F_max_current = F_max / n_counts[0] * n_counts[1]
else:
    F_max_current = F_max

signal = np.concatenate([np.array(50 * [0]),
                         np.linspace(0, F_max_current, n_counts.min()),
                         F_max_current * np.array(abs(n_counts[0] - n_counts[1]) * [1]),
                         np.linspace(F_max_current, 0, n_counts.min()),
                         np.array(50 * [0])]
                        )

# detector impulse response
h_detector_to_convolve = H_detector(np.linspace(0, detector_t, int(detector_t / deltaT_counter)))
# normalized
h_detector_to_convolve = h_detector_to_convolve / sum(h_detector_to_convolve)
# add h_detector
signal_after_detector = np.convolve(signal, h_detector_to_convolve)[:-int(
    detector_t / deltaT_counter) + 1]  # np.array(int(detector_t/ deltaT_counter)*[1])/int(detector_t/ deltaT_counter))

# number of system tics per scan
x_ticks = np.linspace(0, ray_t_pass + detector_t_pass, len(signal_after_detector)) * 10 ** 6

plt.subplot(2, 2, 1)
plt.plot(x_ticks, np.concatenate([50 * [0],
                                  n_counts[1] * [F_max],
                                  (len(x_ticks) - 50 - n_counts[1]) * [0]]))
plt.xlabel('mcs')
plt.ylabel('E')
plt.title('Detector form')

plt.subplot(2, 2, 2)
plt.plot(x_ticks, np.concatenate([50 * [0],
                                  n_counts[0] * [F_max],
                                  (len(x_ticks) - 50 - n_counts[0]) * [0]]))
plt.xlabel('mcs')
plt.ylabel('E')
plt.title('Ray form')
plt.subplot(2, 2, 3)
plt.plot(x_ticks, signal)
plt.xlabel('mcs')
plt.ylabel('E')
plt.title('Signal initial form')
plt.subplot(2, 2, 4)
plt.plot(x_ticks, signal_after_detector)
plt.xlabel('mcs')
plt.ylabel('E')
plt.title('Signal form after detector')
plt.tight_layout()
plt.show()

t_real = x_ticks[50 + int(n_counts[0] / 2) + 1]
t = sum(x_ticks * signal_after_detector) / sum(signal_after_detector)
