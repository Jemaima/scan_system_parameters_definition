import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


deltaT_counter = 20 * 10 ** (-9)  # s
w_rad = 120 * np.pi

initial_object_pos = [0, 0, 4]  # x,y,z

# Distortion coefs
cameraMatrix = np.float64([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]
                          ])

img_size = (int(np.pi / w_rad/deltaT_counter), int(np.pi / w_rad/deltaT_counter))

dist_coefs = np.float64([0, 0, 0, 0])
FILE_NAME = 'object.txt'


# z - camera axix


def get_object_from_file(filename=FILE_NAME):
    f = open(filename, 'r')
    object_transform = np.float64(f.readline()[:-2].split(',')).astype(float)
    object_rotation = np.float64(f.readline()[:-2].split(',')).astype(float)
    object_n_points = int(f.readline())
    object_points = np.zeros([object_n_points, 3])
    for i in range(object_n_points):
        object_points[i] = np.float64(f.readline()[:-2].split(',')).astype(float)
    return TrackedObject(object_points, object_rotation, object_transform)


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
                 rotation_vector=np.array([0.0, 0.0, 0.0]), tranform_vector=np.array([0.0, 0.0, 0.0])):
        UVW_points = np.array(UVW_points)
        self.UVW_points = UVW_points  # UVW_points[UVW_points[:, 2].argsort()]
        self.n_UVW_points = len(UVW_points)
        self.initial_rotation_vector = np.array(rotation_vector)
        self.initial_transform_vector = np.array(tranform_vector)

        # transform parameters
        self.scale = 1.0
        self.rotation_vector = None
        self.transform_vector = None
        self.rotation_matrix = None

        # transform initial object
        self.transformed_UVW_points = None
        # determine visible points and convert them into angles
        self.visible_points = None
        self.n_visible_points = 0
        self.visible_point_coordinates = None

        # cv2.getRotationMatrix2D([0,0],15,2)

    def set_transformation(self, rotation_vector=None, tranыform_vector=None):
        self.rotation_vector = np.float64(rotation_vector)
        self.transform_vector = np.float64(tranыform_vector)
        self.rotation_matrix = generate_rotation_matrix_from_euler_angles(self.rotation_vector)
        self.transformed_UVW_points = np.dot(self.UVW_points, self.rotation_matrix) + initial_object_pos
        #
        self.visible_points = self._find_visible_points() # all points now
        self.n_visible_points = len(self.visible_points)
        self.visible_point_coordinates = self._determine_angles_of_visible_points(self.visible_points)
        self.solvePnP()
        # cv2.solvePnP(self.UVW_points , self.angle_coordinates_of_visible_points, cameraMatrix, dist_coefs)
        # self.uv_ideal_points = self.get_projection()

    def _find_visible_points(self):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        ax.scatter(self.UVW_points.T[0],
                   self.UVW_points.T[1],
                   self.UVW_points.T[2],
                   zdir='z')
        ax.set_title('initial')
        ax2.scatter(self.transformed_UVW_points.T[0],
                    self.transformed_UVW_points.T[1],
                    self.transformed_UVW_points.T[2],
                    zdir='z')
        ax2.set_title('transformed')
        plt.suptitle('Object 3D coordinates')

        mean_z = np.mean(self.transformed_UVW_points.T[2])
        visible_points = self.transformed_UVW_points #[self.transformed_UVW_points.T[2] <= mean_z]

        # left = self.transformed_UVW_points[np.argmin(self.transformed_UVW_points.T[0])]
        # right = self.transformed_UVW_points[np.argmax(self.transformed_UVW_points.T[0])]
        # top = self.transformed_UVW_points[np.argmax(self.transformed_UVW_points.T[1])]
        # bottom = self.transformed_UVW_points[np.argmin(self.transformed_UVW_points.T[1])]
        # center_v = (top + bottom)/2
        # center_h = (left + right) / 2
        # center = [center_v[0],center_h[1]]
        #
        # visible_points = np.array([left, top, right,bottom])
        # for p in self.transformed_UVW_points:
        #     if not any([all(np.equal(p,pp)) for pp in visible_points]):
        #
        #         # fig = plt.figure()
        #         # ax = fig.add_subplot(111, projection='3d')
        #         # ax.scatter(self.transformed_UVW_points.T[0],
        #         #            self.transformed_UVW_points.T[1],
        #         #            self.transformed_UVW_points.T[2],
        #         #            zdir='z')
        #         # ax.scatter(self.UVW_points.T[0],
        #         #            self.UVW_points.T[1],
        #         #            self.UVW_points.T[2],
        #         #            zdir='z')
        #         # ax.scatter(p[0],
        #         #            p[1],
        #         #            p[2],
        #         #            zdir='z', s=40)
        #         # plt.plot([left[0], right[0]],
        #         #          [left[1], right[1]],
        #         #          [left[2], right[2]])
        #         # plt.plot([top[0], bottom[0]],
        #         #          [top[1], bottom[1]],
        #         #          [top[2], bottom[2]])
        #         #
        #         # plt.show()
        #
        #         # left_top
        #         if p[0] <= center[0] and p[1] >= center[1]:
        #             projection = left + (top-left)*(top-p)[1]/(top-left)[1]
        #             if p[2] < projection[2]:
        #                 visible_points = np.append(visible_points, [p],axis=0)
        #         # left_bottom
        #         if p[0] <= center[0] and p[1] <= center[1]:
        #             projection = left + (bottom-left)*(bottom-p)[1]/(bottom-left)[1]
        #             if p[2] < projection[2]:
        #                 visible_points = np.append(visible_points, [p],axis=0)
        #         # right_top
        #         if p[0] >= center[0] and p[1] >= center[1]:
        #             projection = right + (top-right)*(top-p)[1]/(top-right)[1]
        #             if p[2] < projection[2]:
        #                 visible_points = np.append(visible_points, [p],axis=0)
        #         # right_bottom
        #         if p[0] >= center[0] and p[1] <= center[1]:
        #             projection = right + (bottom-right)*(bottom-p)[1]/(bottom-right)[1]
        #             if p[2] < projection[2]:
        #                 visible_points = np.append(visible_points, [p],axis=0)
        return visible_points

    def _determine_angles_of_visible_points(self, points):
        angle_coordinates = np.zeros([len(points), 2])
        for i, p in enumerate(points):
            angle_coordinates[i] = [np.arctan(p[0] / p[2])+np.pi/2, np.arctan(p[1] / p[2])+np.pi/2]
        plt.figure(figsize=(6, 6))
        plt.scatter(angle_coordinates.T[0] * 180 / np.pi, angle_coordinates.T[1] * 180 / np.pi)
        plt.title('angle object projection')
        return (angle_coordinates /w_rad/deltaT_counter).astype(int) #* 180 / np.pi

    def solvePnP(self):
        a = cv2.solvePnP(self.UVW_points, self.visible_point_coordinates, cameraMatrix,
                         dist_coefs, None, None, False, cv2.SOLVEPNP_ITERATIVE)

        cv2.calibrateCameraExtended(self.UVW_points, self._determine_angles_of_visible_points(self.UVW_points), img_size,
                                    cameraMatrix,dist_coefs)
        pass

    def add_noise_to_uv_points(self, noise_range):
        pass


if __name__=='__main__':

    coord = [[-1, -1, -1],
             [-1, 1, -1],
             [1, -1, -1],
             [1, 1, -1],
             [-1, -1, 1],
             [-1, 1, 1],
             [1, -1, 1],
             [1, 1, 1],
             ] * np.random.random([8, 3]) / 10
    cube = TrackedObject(coord)

    cube = get_object_from_file()
    cube.set_transformation([0, 0, 0], [0, 0, 0])
