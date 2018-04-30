import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import object_processing as op

results = pd.read_csv('experiment_logs/results_2018-04-25_22-20-16.csv', decimal=',', sep=';', index_col=0)
results = results.append(pd.read_csv('experiment_logs/results_2018-04-30_11-23-21.csv', decimal=',', sep=';', index_col=0), ignore_index=True)
results = results[results['translation_z']!=7000]
results['restored_rotation_x'] = results['restored_rotation_x'] % 360
results['restored_rotation_y'] = results['restored_rotation_y'] % 360
results['restored_rotation_z'] = results['restored_rotation_z'] % 360
#
# results.loc[(results['rotation_x'] - results['restored_rotation_x']) > 180, 'restored_rotation_x'] = results[
#                                                                                                          'restored_rotation_x'] + 360
# results.loc[(-results['rotation_x'] + results['restored_rotation_x']) > 180, 'restored_rotation_x'] = results[
#                                                                                                           'restored_rotation_x'] - 360
# results.loc[(results['rotation_y'] - results['restored_rotation_y']) > 180, 'restored_rotation_y'] = results[
#                                                                                                          'restored_rotation_y'] + 360
# results.loc[(-results['rotation_y'] + results['restored_rotation_y']) > 180, 'restored_rotation_y'] = results[
#                                                                                                           'restored_rotation_y'] - 360
# results.loc[(results['rotation_z'] - results['restored_rotation_z']) > 180, 'restored_rotation_z'] = results[
#                                                                                                          'restored_rotation_z'] + 360
# results.loc[(-results['rotation_z'] + results['restored_rotation_z']) > 180, 'restored_rotation_z'] = results[
#                                                                                                           'restored_rotation_z'] - 360

results['error_r'] = abs(np.sqrt(np.power(results['rotation_x'] - results['restored_rotation_x'], 2) +
                             np.power(results['rotation_y'] - results['restored_rotation_y'], 2) +
                             np.power(results['rotation_z'] - results['restored_rotation_z'], 2)))

results['error_t'] = abs(np.sqrt(np.power(results['translation_x'] - results['restored_translation_x'], 2) +
                             np.power(results['translation_y'] - results['restored_translation_y'], 2) +
                             np.power(results['translation_z'] - results['restored_translation_z'], 2)))



# ==================================================
# NOISE DEPENDENCE
# ==================================================
noise_dependence = results[results['error_r'] <45].groupby(['noise'])['error_r', 'error_t'].aggregate(np.mean)
# plt.figure()
plt.subplot(2, 1, 1)
# (noise_dependence)['error_r'].plot()
plt.plot(np.tan(noise_dependence.index * op.deltaT_counter)*1000000,noise_dependence['error_r'].values)

plt.ylabel('degree',fontsize = 14)
plt.title('rotation error',fontsize = 14)
plt.xlabel('noise, mсs',fontsize = 14)
plt.subplot(2, 1, 2)
plt.plot(np.tan(noise_dependence.index * op.deltaT_counter)*1000000,noise_dependence['error_t'].values)
plt.plot(np.tan(noise_dependence.index * op.deltaT_counter)*1000000,[3]*len(noise_dependence['error_t'].values))
plt.ylabel('mm',fontsize = 14)
plt.title('translation error',fontsize = 14)
plt.xlabel('noise, mcs',fontsize = 14)
plt.tight_layout()

# ==================================================
# AREA DEPENDENCE FOR IDEAL SYSTEM
# ==================================================
# mean error on y lvl = 0 noise = 0
plt.figure(figsize=(7, 7))
area_rot_error_dependence = \
results[results['noise'] == 0.0][results['translation_y'] == 0.0].groupby(['translation_x', 'translation_z'])[
    'error_t'].aggregate(np.mean).unstack()
plt.imshow(area_rot_error_dependence, interpolation='nearest')
plt.ylabel('x,mm',fontsize = 12)
plt.xlabel('distance,mm',fontsize = 12)
plt.colorbar()
plt.clim(vmin=0.0)
plt.yticks(np.arange(len(area_rot_error_dependence.index.values)), area_rot_error_dependence.index.values.astype(int))
plt.xticks(np.arange(len(area_rot_error_dependence.columns.values)), area_rot_error_dependence.columns.values.astype(int))
plt.suptitle('translation error')

plt.figure(figsize=(7, 7))
area_rot_error_dependence = \
results[results['noise'] == 0.0][results['translation_y'] == 0.0][results['error_r'] <45].groupby(['translation_x', 'translation_z'])[
    'error_r'].aggregate(np.mean).unstack()
plt.imshow(area_rot_error_dependence, interpolation='nearest')
plt.ylabel('x,mm',fontsize = 12)
plt.xlabel('distance,mm',fontsize = 12)
plt.colorbar()
plt.clim(vmin=0.0)
plt.yticks(np.arange(len(area_rot_error_dependence.index.values)), area_rot_error_dependence.index.values.astype(int))
plt.xticks(np.arange(len(area_rot_error_dependence.columns.values)), area_rot_error_dependence.columns.values.astype(int))
plt.suptitle('rotation error')


# ==================================================
# AREA DEPENDENCE FOR REAL SYSTEM
# ==================================================

noise_values = results.noise.unique()
plt.figure(figsize=(7, 5))
i = 1
for n in noise_values[0:4]:
    plt.subplot(2, 2, 5-i)
    area_rot_error_dependence = \
        results[results['noise'] == n][results['translation_y'] == 0.0][results['translation_z'] < 6000].groupby(['translation_x', 'translation_z'])[
            'error_t'].aggregate(np.mean).unstack()
    plt.imshow(area_rot_error_dependence, interpolation='nearest')
    plt.ylabel('x,mm',fontsize = 12)
    plt.xlabel('distance,mm',fontsize = 12)
    plt.colorbar()
    plt.clim(vmin=0.0)
    plt.yticks(np.arange(len(area_rot_error_dependence.index.values)), area_rot_error_dependence.index.values.astype(int))
    plt.xticks(np.arange(len(area_rot_error_dependence.columns.values)), area_rot_error_dependence.columns.values.astype(int))
    i = i + 1
    plt.title(' '.join(['noise =',"{:4.1f}".format(n * op.deltaT_counter*1000000),'mcs']),fontsize = 14)
plt.tight_layout()


plt.figure(figsize=(7, 5))
i = 1
for n in noise_values[0:4]:
    plt.subplot(2, 2, 5-i)
    area_rot_error_dependence = \
        results[results['noise'] == n][results['translation_y'] == 0.0][results['translation_z'] < 6000][results['error_r'] <45].groupby(['translation_x', 'translation_z'])[
            'error_r'].aggregate(np.mean).unstack()
    plt.imshow(area_rot_error_dependence, interpolation='nearest')
    plt.ylabel('x,mm',fontsize = 12)
    plt.xlabel('distance,mm',fontsize = 12)
    plt.colorbar()
    plt.clim(vmin=0.0)
    plt.yticks(np.arange(len(area_rot_error_dependence.index.values)), area_rot_error_dependence.index.values.astype(int))
    plt.xticks(np.arange(len(area_rot_error_dependence.columns.values)), area_rot_error_dependence.columns.values.astype(int))
    i = i + 1
    plt.title(' '.join(['noise =',"{:4.1f}".format(n * op.deltaT_counter*1000000),'mcs']),fontsize = 14)

plt.tight_layout()
plt.show()