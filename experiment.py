import numpy as np
import itertools
import matplotlib.pyplot as plt
import object_processing as op
import cv2
import pandas as pd
import datetime

cube = op.get_object_from_file(initial_transform_vector=op.initial_object_pos)

results = pd.DataFrame(columns=['noise',
                                'rotation_x', 'rotation_y', 'rotation_z',
                                'translation_x', 'translation_y', 'translation_z',
                                'restored_rotation_x', 'restored_rotation_y', 'restored_rotation_z',
                                'restored_translation_x', 'restored_translation_y', 'restored_translation_z'])

noise_scale_to_test = [100,50,30,10,0] #, 0.0001, 0.001, 0.01, 0.1]
angles_to_test = np.array([a for a in itertools.product(np.linspace(0,180,5), np.linspace(0,180,5),np.linspace(0,180,5))])
L_to_test = np.array([500,1000,3000,5000,7000]) #np.linspace(500,6000,12)
poper_to_test = np.array([t for t in itertools.product(np.linspace(-2000,2000, 5), np.linspace(-2000,2000, 5))])

translates_to_test = np.concatenate([np.repeat(poper_to_test,len(L_to_test),axis=0),
                                    np.expand_dims(np.array([L_to_test] * len(poper_to_test)).flatten(), 1)],
                                   axis=1)

# cube = get_object_from_file()
for n in noise_scale_to_test:
    for r in angles_to_test:
        for t in translates_to_test:
            cube.set_transformation(r, t, n)
            r_restored, t_restored = cube.restore_position()

            results = results.append({'noise': n,
                                      'rotation_x': r[0], 'rotation_y': r[1], 'rotation_z': r[2],
                                      'translation_x': t[0], 'translation_y': t[1], 'translation_z': t[2],
                                      'restored_rotation_x': r_restored[0], 'restored_rotation_y': r_restored[1],
                                      'restored_rotation_z': r_restored[2],
                                      'restored_translation_x': t_restored[0], 'restored_translation_y': t_restored[1],
                                      'restored_translation_z': t_restored[2]},
                                     ignore_index=True)
            # print('R: \t\t\t', ", ".join("%.2f" % f for f in (r_restored)))
            # print('R initial: \t', ", ".join("%.2f" % f for f in cube.rotation_vector))
            # print('T: \t\t\t', ", ".join("%.2f" % f for f in t_restored ))
            # print('T initial: \t', ", ".join("%.2f" % f for f in cube.transform_vector))

    print('noise_lvl ',str(n))

results.to_csv('logs\\results_'+ (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '.csv', decimal=',', sep=';')
