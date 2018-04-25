import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

results = pd.read_csv( 'logs\\results_2018-04-25_02-03-30.csv', decimal=',', sep=';',index_col=0)
results['restored_rotation_x'] = results['restored_rotation_x']%360
results['restored_rotation_y'] = results['restored_rotation_y']%360
results['restored_rotation_z'] = results['restored_rotation_z']%360

results.loc[(results['rotation_x']-results['restored_rotation_x']) > 180, 'restored_rotation_x']=results['restored_rotation_x']+360
results.loc[(-results['rotation_x']+results['restored_rotation_x']) > 180, 'restored_rotation_x']=results['restored_rotation_x']-360
results.loc[(results['rotation_y']-results['restored_rotation_y']) > 180, 'restored_rotation_y']=results['restored_rotation_y']+360
results.loc[(-results['rotation_y']+results['restored_rotation_y']) > 180, 'restored_rotation_y']=results['restored_rotation_y']-360
results.loc[(results['rotation_z']-results['restored_rotation_z']) > 180, 'restored_rotation_z']=results['restored_rotation_z']+360
results.loc[(-results['rotation_z']+results['restored_rotation_z']) > 180, 'restored_rotation_z']=results['restored_rotation_z']-360


results['error_r'] = np.sqrt(np.power(results['rotation_x']-results['restored_rotation_x'],2) +
                             np.power(results['rotation_y']-results['restored_rotation_y'],2) +
                             np.power(results['rotation_z']-results['restored_rotation_z'],2))-162.5

results['error_t'] = np.sqrt(np.power(results['translation_x']-results['restored_translation_x'],2) +
                             np.power(results['translation_y']-results['restored_translation_y'],2) +
                             np.power(results['translation_z']-results['restored_translation_z'],2))

noise_dependence = results.groupby(['noise'])['error_r','error_t'].aggregate(np.mean)
# plt.figure()
plt.subplot(2,1,1)
(noise_dependence)['error_r'].plot()
plt.ylabel('degree')
plt.title('rotation error')
plt.xlabel('noise, ticks')
plt.subplot(2,1,2)
(noise_dependence)['error_t'].plot()
plt.ylabel('mm')
plt.title('translation error')
plt.xlabel('noise, ticks')
plt.tight_layout()

dist_dependence = results[results['noise']==0.0].groupby(['translation_z'])['error_r','error_t'].aggregate(np.mean)
plt.subplot(2,1,1)
(dist_dependence)['error_r'].plot()
plt.ylabel('degree')
plt.title('rotation error')
plt.xlabel('distance, m')
plt.subplot(2,1,2)
(dist_dependence)['error_t'].plot()
plt.ylabel('mm')
plt.title('translation error')
plt.xlabel('distance, m')
plt.tight_layout()

x_dependence = results[results['noise']==0.0].groupby(['translation_x'])['error_r','error_t'].aggregate(np.mean)
plt.subplot(2,1,1)
(x_dependence)['error_r'].plot()
plt.ylabel('degree')
plt.title('rotation error')
plt.xlabel('x, mm')
plt.ylim([0,3])
plt.subplot(2,1,2)
(x_dependence)['error_t'].plot()
plt.ylabel('mm')
plt.title('translation error')
plt.xlabel('x, mm')
plt.tight_layout()


angle_dependence = results.groupby(['rotation_x','rotation_y','rotation_z'])['error_r','error_t'].aggregate(np.mean)
plt.subplot(2,1,1)
(angle_dependence)['error_r'].plot(kind = 'bar')
plt.ylabel('degree')
plt.title('rotation error')
plt.xlabel('distance, m')
plt.subplot(2,1,2)
(angle_dependence)['error_t'].plot(kind = 'bar')
plt.ylabel('mm')
plt.title('translation error')
plt.tight_layout()


fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface()

area_depengence = results[results['translation_y']==0].groupby(['translation_z','translation_x'])['error_r','error_t'].aggregate(np.mean)
