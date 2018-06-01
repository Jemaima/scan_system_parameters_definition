import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import object_processing as op

results = pd.read_csv('signal_logs/results_2018-04-30_15-15-47.csv', decimal=',', sep=';', index_col=0)

# ==================================================
# WIDTH DEPENDENCE
# ==================================================
width_dependence_max = results.groupby(['width', 'noise'])['t_error'].aggregate(np.max).unstack()
# width_dependence_max[width_dependence_max>1.5] =0.0
plt.figure()
plt.subplot(1, 2, 1)
# (noise_dependence)['error_r'].plot()
plt.imshow(width_dependence_max, interpolation='nearest')
plt.colorbar()
plt.clim(vmin=0.0)
plt.xlabel('SNR')
plt.ylabel('ray width, mm')
plt.title('max time error, mcs')
plt.yticks(np.arange(len(width_dependence_max.index.values)), width_dependence_max.index.values.astype(int))
plt.xticks(np.arange(len(width_dependence_max.columns.values)),
           np.round(1 / width_dependence_max.columns.values,2))


width_dependence_mean = results.groupby(['width', 'noise'])['t_error'].aggregate(np.mean).unstack()
plt.subplot(1, 2, 2)
# (noise_dependence)['error_r'].plot()
plt.imshow(width_dependence_mean, interpolation='nearest')
plt.colorbar()
plt.clim(vmin=0.0)
plt.xlabel('SNR')
plt.ylabel('ray width, mm')
plt.title('mean time error, mcs')
plt.yticks(np.arange(len(width_dependence_mean.index.values)), width_dependence_mean.index.values.astype(int))
plt.xticks(np.arange(len(width_dependence_mean.columns.values)),
           np.round(1 / width_dependence_mean.columns.values, 2))


plt.show()
