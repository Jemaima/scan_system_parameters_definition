import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

deltaT_counter = 20 * 10 ** (-9)  # s
w_rad = 120 * np.pi

initial_object_pos = np.array([0, 0, 1000], np.float32)  # x,y,z , mm

img_size = (int(np.pi / w_rad / deltaT_counter), int(np.pi / w_rad / deltaT_counter))

cameraMatrix = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]
                         ], dtype=np.float64)

dist_coefs = np.float64([0, 0, 0, 0, 0])
FILE_NAME = 'object.txt'


# z - camera axix


def get_object_from_file(filename=FILE_NAME,  initial_transform_vector=initial_object_pos):
    f = open(filename, 'r')
    object_transform = np.float64(f.readline()[:-2].split(',')).astype(float)
    object_rotation = np.float64(f.readline()[:-2].split(',')).astype(float)
    object_n_points = int(f.readline())
    object_points = np.zeros([object_n_points, 3])
    for i in range(object_n_points):
        object_points[i] = np.float64(f.readline()[:-2].split(',')).astype(float)
    return TrackedObject(50*object_points, object_rotation, object_transform + initial_transform_vector)


def generate_rotation_matrix_from_euler_angles(euler_angles):
    euler_angles_rad = euler_angles * np.pi / 180
    sx, cx = np.sin(euler_angles_rad[0]), np.cos(euler_angles_rad[0])
    sy, cy = np.sin(euler_angles_rad[1]), np.cos(euler_angles_rad[1])
    sz, cz = np.sin(euler_angles_rad[2]), np.cos(euler_angles_rad[2])

    R_x = np.array([[1, 0, 0],
                    [0, cx, -sx],
                    [0, sx, cx]
                    ])

    R_y = np.array([[cy, 0, sy],
                    [0, 1, 0],
                    [-sy, 0, cy]
                    ])

    R_z = np.array([[cz, -sz, 0],
                    [sz, cz, 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


class TrackedObject(object):
    def __init__(self, UVW_points,
                 initial_rotation_vector=np.array([0.0, 0.0, 0.0]),
                 initial_transform_vector=np.array([0.0, 0.0, 0.0])
                 ):

        self.UVW_points = np.array(UVW_points, np.float32)  # UVW_points[UVW_points[:, 2].argsort()]
        self.n_UVW_points = len(UVW_points)
        self.initial_rotation_vector = np.array(initial_rotation_vector)
        self.initial_transform_vector = np.array(initial_transform_vector)

        # transform parameters
        self.scale = 1.0
        self.rotation_vector = None
        self.transform_vector = None
        self.rotation_matrix = None

        # transform initial object
        self.transformed_UVW_points = None
        # determine visible points and convert them into angles
        self.angle_points_projection = None
        self.n_visible_points = 0
        self.points_projection = None
        self.visible_point_coordinates = None

        # cv2.getRotationMatrix2D([0,0],15,2)

    def set_transformation(self, rotation_vector=None, transform_vector=None, noise_scale =0.0):
        self.rotation_vector = np.float64(rotation_vector)
        self.transform_vector = np.float64(transform_vector)

        self.rotation_matrix = generate_rotation_matrix_from_euler_angles(self.rotation_vector)

        self.transformed_UVW_points = np.dot(self.UVW_points, self.rotation_matrix) \
                                      + transform_vector + self.initial_transform_vector

        # self.angle_points_projection = self.points2angles(self.transformed_UVW_points, True)
        self.points_projection = np.squeeze(cv2.projectPoints(self.UVW_points,
                                                              self.rotation_vector / 180 * np.pi,
                                                              self.transform_vector + self.initial_transform_vector,
                                                              cameraMatrix,
                                                              dist_coefs)[0], 1)

        self.points_projection_noise = self.points_projection +\
                                       np.random.random(self.points_projection.shape) * \
                                       noise_scale * (self.points_projection.max() - self.points_projection.min())

        # self.visible_point_coordinates = self.points2angles(self.transformed_UVW_points, False)
        # self.solvePnP()
        # cv2.solvePnP(self.UVW_points , self.angle_coordinates_of_visible_points, cameraMatrix, dist_coefs)
        # self.uv_ideal_points = self.get_projection()

    #TODO
    def _find_visible_points(self):
        visible_points = self.transformed_UVW_points  # [self.transformed_UVW_points.T[2] <= mean_z]
        return visible_points

    def points2angles(self, points, with_z=False):
        if with_z:
            angle_coordinates = np.zeros([len(points), 3])
            for i, p in enumerate(points):
                angle_coordinates[i] = [np.arctan(p[0] / p[2]) + np.pi / 2, np.arctan(p[1] / p[2]) + np.pi / 2, p[2]]

        else:
            angle_coordinates = np.zeros([len(points), 2])
            for i, p in enumerate(points):
                angle_coordinates[i] = [np.arctan(p[0] / p[2]) + np.pi / 2, np.arctan(p[1] / p[2]) + np.pi / 2]
                # angle_coordinates = angle_coordinates/ w_rad / deltaT_counter

        # angle_coordinates[:, :2] = angle_coordinates[:, :-1] / w_rad / deltaT_counter
        return angle_coordinates  # * 180 / np.pi

    def angles2points(self, angles):
        if angles.shape[1] == 2:
            return None

        points = np.zeros([len(angles), 3])
        for i, p in enumerate(angles):
            points[i] = np.tan(p * deltaT_counter * w_rad - np.pi / 2) * p[2]
        plt.figure(figsize=(6, 6))
        plt.scatter(points.T[0], points.T[1])
        plt.title('angle object projection')
        return points


if __name__ == '__main__':
    coord = [[-1, -1, -1],
             [-1,  1, -1],
             [ 1, -1, -1],
             [ 1,  1, -1],
             [-1, -1,  1],
             [-1,  1,  1],
             [ 1, -1,  1],
             [ 1,  1,  1],
             ]  # * np.random.random([8, 3]) / 10
    cube = TrackedObject(coord, initial_transform_vector=initial_object_pos)
    cube = get_object_from_file(initial_transform_vector=initial_object_pos)

    # cube = get_object_from_file()
    cube.set_transformation([120, 70, 45], [0, 0, 0], 0.00001)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    ax.scatter(cube.UVW_points.T[0],
               cube.UVW_points.T[1],
               cube.UVW_points.T[2],
               zdir='z')
    ax.set_title('initial')
    ax2.scatter(cube.transformed_UVW_points.T[0],
                cube.transformed_UVW_points.T[1],
                cube.transformed_UVW_points.T[2],
                zdir='z')
    ax2.set_title('transformed')
    plt.suptitle('Object 3D coordinates')

    plt.figure(figsize=(6, 6))
    plt.scatter(cube.points_projection.T[0], cube.points_projection.T[1])
    plt.title('object projection through cv2.projectPoints')
    plt.show()

    # cube.points_projection_noise = cube.points_projection + np.random.random(cube.points_projection.shape) / 10
    _, r_restored, t_restored = cv2.solvePnP(cube.UVW_points,
                                             cube.points_projection_noise.astype(np.float32),
                                             cameraMatrix,
                                             dist_coefs,
                                             tvec=initial_object_pos,
                                             flags = cv2.SOLVEPNP_ITERATIVE)
    r_restored, t_restored = np.squeeze(r_restored), np.squeeze(t_restored)/6
    print('R: \t\t\t', ", ".join("%.2f" % f for f in (r_restored / np.pi * 180)))
    print('R initial: \t', ", ".join("%.2f" % f for f in cube.rotation_vector))
    print('T: \t\t\t', ", ".join("%.3f" % f for f in t_restored / np.pi * 180))
    print('T initial: \t', ", ".join("%.2f" % f for f in cube.transform_vector + cube.initial_transform_vector))
