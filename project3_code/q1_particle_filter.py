import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from read_data import read_world, read_sensor_data


def get_sensor_trajectory(sensor_data):
    num_of_samples = int(len(sensor_data.keys()) / 2)
    x = [0]
    y = [0]
    theta = [0]
    for ii in range(num_of_samples):
        delta_rot_1 = sensor_data[(ii, 'odometry')]['r1']
        delta_rot_2 = sensor_data[(ii, 'odometry')]['r2']
        delta_trans = sensor_data[(ii, 'odometry')]['t']

        x.append(x[-1] + delta_trans * np.cos(theta[-1] + delta_rot_1))
        y.append(y[-1] + delta_trans * np.sin(theta[-1] + delta_rot_1))
        theta.append(theta[-1] + delta_rot_1 + delta_rot_2)
    trajectory = np.vstack([x, y, theta]).T

    return trajectory


def particle_filter(landmarks, world_data, sensor_data):
    gt_trajectory = get_sensor_trajectory(sensor_data)

    if False:
        plt.figure()
        plt.scatter(gt_trajectory[:,0], gt_trajectory[:,1])
        plt.show(block=False)

    return


def main():
    landmarks_fp = r'/home/nadav/studies/mapping_and_perception_autonomous_robots/project_3/landmarks_EX1.csv'
    world_data_fp = r'/home/nadav/studies/mapping_and_perception_autonomous_robots/project_3/Project3_ex2/world.dat'
    sensor_data_fp = r'/home/nadav/studies/mapping_and_perception_autonomous_robots/project_3/Project3_ex2/sensor_data.dat'

    landmarks = np.array(pd.read_csv(landmarks_fp, header=None))
    world_data = read_world(world_data_fp)
    sensor_data = read_sensor_data(sensor_data_fp)

    if False:
        plt.figure()
        plt.scatter(landmarks[:, 0], landmarks[:, 1])
        plt.show(block=False)
    particle_filter(landmarks, world_data, sensor_data)


if __name__ == "__main__":
    main()
