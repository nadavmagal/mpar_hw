# % This is the main extended Kalman filter SLAM loop. This script calls all the required
# % functions in the correct order.
# %
# % You can disable the plotting or change the number of steps the filter
# % runs for to ease the debugging. You should however not change the order
# % or calls of any of the other lines, as it might break the framework.
# %
# % If you are unsure about the input and return values of functions you
# % should read their documentation which tells you the expected dimensions.

# % Turn off pagination:
# more off;

# % clear all variables and close all windows
# figure('visible', 'on');
# clear all;
# close all;

# % Make tools available
# addpath('tools');
import os
import numpy as np
from matplotlib import pyplot as plt
from q3_efk_slam.tools import *
from q3_efk_slam.prediction_step import prediction_step
from q3_efk_slam.correction_step import correction_step


def get_gt(data):
    total_gt = []
    cur_gt = np.array([0., 0., 0.])
    total_gt.append(cur_gt)
    for cur_timestep in data['timestep']:
        cur_t = cur_timestep['odometry']['t']
        cur_r1 = cur_timestep['odometry']['r1']
        cur_r2 = cur_timestep['odometry']['r2']
        prev_theta = cur_gt[2]

        cur_gt[0] += cur_t * np.cos(prev_theta + cur_r1)
        cur_gt[1] += cur_t * np.sin(prev_theta + cur_r1)
        cur_gt[2] += cur_r1 + cur_r2
        total_gt.append(cur_gt.copy())
    total_gt = np.array(total_gt)
    return total_gt


def ekf_slam_func(data_path):
    # % Read world data, i.e. landmarks. The true landmark positions are not given to the robot
    landmarks = read_world(os.path.join(data_path, 'world.dat'))

    # % load landmarks;
    # % Read sensor readings, i.e. odometry and range-bearing sensor
    data = read_data(os.path.join(data_path, 'sensor_data.dat'))
    # TODO: do we have the delta r1 r2 t or we have the absolute values
    if False:
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot([cur_timestep['odometry']['r1'] for cur_timestep in data['timestep']])

        plt.subplot(3, 1, 2)
        plt.plot([cur_timestep['odometry']['r2'] for cur_timestep in data['timestep']])

        plt.subplot(3, 1, 3)
        plt.plot([cur_timestep['odometry']['t'] for cur_timestep in data['timestep']])

    INF = 1000
    # % Get the number of landmarks in the map
    N = landmarks.shape[0]

    '''
    # % observedLandmarks is a vector that keeps track of which landmarks have been observed so far.
    # % observedLandmarks(i) will be true if the landmark with id = i has been observed at some point by the robot
    # observedLandmarks = repmat(false,1,N);
    '''
    observedLandmarks = np.array([False] * N)

    # % TODO Initialize belief:
    '''
    # % mu: 2N+3x1 vector representing the mean of the normal distribution
    # % The first 3 components of mu correspond to the pose of the robot,
    # % and the landmark poses (xi, yi) are stacked in ascending id order.
    # % sigma: (2N+3)x(2N+3) covariance matrix of the normal distribution
    '''
    x0, y0, theta0 = 0., 0., 0.
    mu = np.hstack([np.array([x0, y0, theta0]), np.zeros(2 * N)])

    sigma_x0, sigma_y0, sigma_theta0 = 0., 0., 0.  # TODO - think how to initialize
    sigma_xx_0 = np.diag([sigma_x0 ** 2, sigma_y0 ** 2, sigma_theta0 ** 2])
    sigma_mm_0 = np.diag([INF] * 2 * N)
    sigma_xm_0 = np.zeros([3, 2 * N])
    sigma_mx_0 = np.zeros([2 * N, 3])
    sigma = np.vstack([np.hstack([sigma_xx_0, sigma_xm_0]), np.hstack([sigma_mx_0, sigma_mm_0])])
    a = 3

    # % toogle the visualization type
    # %showGui = true;  % show a window while the algorithm runs
    # showGui = false; % plot to files instead

    # % Perform filter update for each odometry-observation pair read from the
    # % data file.
    # for t = 1:size(data.timestep, 2)
    total_mu = []
    total_sigma = []
    for t in range(len(data['timestep'])):
        # % Perform the prediction step of the EKF
        a = 3

        mu, sigma = prediction_step(mu, sigma, data['timestep'][t]['odometry'])

        # % Perform the correction step of the EKF
        mu, sigma, observedLandmarks = correction_step(mu, sigma, data['timestep'][t]['sensor'], observedLandmarks)

        total_mu.append(mu)
        total_sigma.append(sigma)

    total_mu = np.array(total_mu)
    total_sigma = np.array(total_sigma)

    gt_xytheta = get_gt(data)

    plt.figure()
    plt.plot(gt_xytheta[:,0], gt_xytheta[:,1])
    plt.plot(total_mu[:, 0], total_mu[:, 1])
    plt.scatter(landmarks[:, 1], landmarks[:, 2], s=3, color='black')
    plt.show(block=False)

    a = 3


def main():
    data_path = '/home/nadav/studies/mapping_and_perception_autonomous_robots/project_2/Ex3_verB'
    ekf_slam_func(data_path)


if __name__ == "__main__":
    main()
