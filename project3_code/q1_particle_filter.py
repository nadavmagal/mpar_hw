import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from read_data import read_world, read_sensor_data


def get_odometry_trajectory(odometry):
    num_of_samples = odometry.shape[0]
    x = [0]
    y = [0]
    theta = [0]
    for ii in range(num_of_samples):
        # delta_rot_1 = sensor_data[(ii, 'odometry')]['r1']
        # delta_rot_2 = sensor_data[(ii, 'odometry')]['r2']
        # delta_trans = sensor_data[(ii, 'odometry')]['t']
        delta_rot_1 = odometry[ii, 0]
        delta_rot_2 = odometry[ii, 2]
        delta_trans = odometry[ii, 1]

        x.append(x[-1] + delta_trans * np.cos(theta[-1] + delta_rot_1))
        y.append(y[-1] + delta_trans * np.sin(theta[-1] + delta_rot_1))
        theta.append(theta[-1] + delta_rot_1 + delta_rot_2)
    trajectory = np.vstack([x, y, theta]).T

    return trajectory


def particle_filter(landmarks, odometry):
    gt_trajectory = get_odometry_trajectory(odometry)

    if True:
        plt.figure()
        plt.scatter(gt_trajectory[:,0], gt_trajectory[:,1],color='black', label='GT trajectory')
        plt.scatter(landmarks[:, 0], landmarks[:, 1],color='blue', label='landmarks')
        plt.legend()
        plt.show(block=False)
    return


def main():
    landmarks_fp = r'/home/nadav/studies/mapping_and_perception_autonomous_robots/project_3/ParticleEX1/landmarks_EX1.csv'
    odometry_fp = r'/home/nadav/studies/mapping_and_perception_autonomous_robots/project_3/ParticleEX1/odometry.dat'

    landmarks = np.array(pd.read_csv(landmarks_fp, header=None))
    odometry = np.array(pd.read_csv(odometry_fp, header=None, delimiter=' '))[:,1:] #r1, trans, r2

    particle_filter(landmarks, odometry)


if __name__ == "__main__":
    main()
