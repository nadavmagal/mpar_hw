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
from matplotlib.patches import Ellipse

from q3_efk_slam.tools import *
from q3_efk_slam.prediction_step import prediction_step
from q3_efk_slam.correction_step import correction_step
import copy
from common import get_maxe_rmse, save_video_frame
import time
from frames_to_video import create_video

SAVE_VIDEO = False
vid_fig_slam, vid_axs_slam = plt.subplots(1, 1, figsize=[15, 10], dpi=300)


def get_trajectory(data):
    total_gt = []
    cur_gt = np.array([0., 0., 0.])
    total_gt.append(cur_gt.copy())
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


def ekf_slam_func(data_path, result_dir_timed):
    # % Read world data, i.e. landmarks. The true landmark positions are not given to the robot
    landmarks = read_world(os.path.join(data_path, 'world.dat'))

    # % load landmarks;
    # % Read sensor readings, i.e. odometry and range-bearing sensor
    noise = {
        'sigme_rot1': 0,
        'sigme_t': 0,
        'sigme_rot2': 0}
    gt_data = read_data(os.path.join(data_path, 'sensor_data.dat'), noise)

    # noise = {
    #     'sigme_rot1': 0.1,
    #     'sigme_t': 0.2,
    #     'sigme_rot2': 0.1}
    noise = {
        'sigme_rot1': 0.1,
        'sigme_t': 0.2,
        'sigme_rot2': 0.1}
    noised_data = read_data(os.path.join(data_path, 'sensor_data.dat'), noise)
    gt_xytheta = get_trajectory(gt_data)
    noised_xytheta = get_trajectory(noised_data)
    if True:
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot([cur_timestep['odometry']['r1'] for cur_timestep in gt_data['timestep']])

        plt.subplot(3, 1, 2)
        plt.plot([cur_timestep['odometry']['r2'] for cur_timestep in gt_data['timestep']])

        plt.subplot(3, 1, 3)
        plt.plot([cur_timestep['odometry']['t'] for cur_timestep in gt_data['timestep']])

        plt.figure()
        plt.scatter(gt_xytheta[:, 0], gt_xytheta[:, 1], color='black', s=3, label='GT')
        plt.scatter(noised_xytheta[:, 0], noised_xytheta[:, 1], color='red', s=3, label='noised')
        plt.scatter(landmarks[:, 1], landmarks[:, 2], s=5, color='black', marker='<', label='landmarkd')
        plt.title('GT and noised trajectory ')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.grid()
        plt.legend()
        plt.show(block=False)

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
    mu = np.hstack([np.array([x0, y0, theta0]), np.zeros(2 * N)]).copy()

    sigma_x0, sigma_y0, sigma_theta0 = 1., 1., 1.  # TODO - think how to initialize
    sigma_xx_0 = np.diag([sigma_x0 ** 2, sigma_y0 ** 2, sigma_theta0 ** 2])
    sigma_mm_0 = np.diag([INF] * 2 * N)
    sigma_xm_0 = np.zeros([3, 2 * N])
    sigma_mx_0 = np.zeros([2 * N, 3])
    sigma = np.vstack([np.hstack([sigma_xx_0, sigma_xm_0]), np.hstack([sigma_mx_0, sigma_mm_0])]).copy()
    a = 3

    # % toogle the visualization type
    # %showGui = true;  % show a window while the algorithm runs
    # showGui = false; % plot to files instead

    # % Perform filter update for each odometry-observation pair read from the
    # % data file.
    # for t = 1:size(data.timestep, 2)

    sigmot = {
        'sigma_rot1': 0.1,
        'sigma_t': 0.2,
        'sigma_rot2': 0.1,
        'sigma_r_squar': 0.5,
        'sigma_phi_squar': 0.3
    }

    total_mu = []
    total_sigma = []
    total_mu.append(mu.copy())
    total_sigma.append(sigma.copy())
    # data = gt_data
    data = noised_data
    for t in range(len(data['timestep'])):
        # % Perform the prediction step of the EKF
        print(f'{t}/{len(data["timestep"])}')

        mu, sigma = prediction_step(mu.copy(), sigma.copy(), data['timestep'][t]['odometry'], sigmot)

        # % Perform the correction step of the EKF
        mu, sigma, observedLandmarks = correction_step(mu.copy(), sigma.copy(), data['timestep'][t]['sensor'],
                                                       observedLandmarks, sigmot)

        total_mu.append(mu.copy())
        total_sigma.append(sigma.copy())

        if SAVE_VIDEO:
            total_mu_arr = np.array(total_mu)
            plt.figure(vid_fig_slam)
            ax0 = vid_axs_slam
            ax0.clear()
            ax0.scatter(gt_xytheta[:t, 0], gt_xytheta[:t, 1], s=1, color='black', label='GT')
            ax0.scatter(landmarks[:, 1], landmarks[:, 2], s=5, color='black', marker='<', label='landmarkd')
            plt.scatter(total_mu_arr[:, 0], total_mu_arr[:, 1], s=2, color='blue', label='slam_tr')

            # ellipse for robot
            cur_ellipse = Ellipse((mu[0], mu[1]), sigma[0][0], sigma[1][1],
                                  np.rad2deg(mu[2]), edgecolor='green',
                                  fc='None', lw=1)
            ax0.add_patch(cur_ellipse)

            for kk, is_observed in enumerate(observedLandmarks):
                if is_observed == False:
                    continue
                ax0.scatter(landmarks[kk, 1], landmarks[kk, 2], s=5, color='orange', marker='<')
                cur_ellipse = Ellipse((mu[2 * kk + 3], mu[2 * kk + 4]), sigma[2 * kk + 3][2 * kk + 3],
                                      sigma[2 * kk + 4][2 * kk + 4],
                                      np.rad2deg(0), edgecolor='orange', fc='None', lw=1)
                ax0.add_patch(cur_ellipse)

                a = 3

            plt.xlim([-5, 15])
            plt.ylim([-5, 15])
            ax0.legend()
            ax0.grid()

            image_path = os.path.join(result_dir_timed, f'{t}.png')
            vid_fig_slam.savefig(image_path, dpi=150)

        if False:
            plt.figure()
            total_mu = np.array(total_mu)
            total_sigma = np.array(total_sigma)
            plt.scatter(total_mu[:, 0], total_mu[:, 1], s=2, color='blue', label='slam_tr')

            plt.scatter(total_mu[:, 3], total_mu[:, 4], s=2, color='orange', label='slam_lm')
            plt.text(total_mu[-1, 3], total_mu[-1, 4], '1 s', color='orange')
            plt.scatter(total_mu[:, 5], total_mu[:, 6], s=2, color='magenta', label='slam_lm')
            plt.text(total_mu[-1, 5], total_mu[-1, 6], '2 s', color='magenta')

            plt.scatter(gt_xytheta[:, 0], gt_xytheta[:, 1], s=2, color='black', label='gt')
            plt.scatter(landmarks[:, 1], landmarks[:, 2], s=5, color='black', marker='<', label='gt_lm')
            plt.text(landmarks[0, 1], landmarks[0, 2], '1 g')
            plt.text(landmarks[1, 1], landmarks[1, 2], '2 g')

            plt.legend()
            plt.xlim([-5, 15])
            plt.ylim([-5, 15])
            plt.show(block=False)
            plt.close('all')

            total_mu = list(total_mu)
            total_sigma = list(total_sigma)

    total_mu = np.array(total_mu)
    total_sigma = np.array(total_sigma)

    max_E, rmse = get_maxe_rmse(gt_xytheta, total_mu)

    plt.figure()
    plt.scatter(gt_xytheta[:, 0], gt_xytheta[:, 1], color='black', s=3, label='GT')
    # plt.scatter(noised_xytheta[:, 0], noised_xytheta[:, 1], color='red', s=3, label='noised')
    plt.scatter(landmarks[:, 1], landmarks[:, 2], s=5, color='black', marker='<', label='landmarkd')
    plt.scatter(total_mu[:, 0], total_mu[:, 1], s=2, color='blue', label='slam_tr')
    plt.title(f'EKF-SLAM results - rmse={round(rmse, 2)}, maxE={round(max_E, 2)}')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim([-5, 15])
    plt.ylim([-5, 15])
    plt.grid()
    plt.legend()
    plt.show(block=False)

    ''' plot cov relations '''
    diff_from_gt = gt_xytheta - total_mu[:, 0:3]
    diff_theta = np.where(diff_from_gt[:, 2] < -np.pi, diff_from_gt[:, 2] + 2 * np.pi, diff_from_gt[:, 2])

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(diff_from_gt[:, 0], label='error in x [m]')
    plt.plot(np.sqrt(total_sigma[:, 0, 0]), color='red', label='sigma x')
    plt.plot(-np.sqrt(total_sigma[:, 0, 0]), color='red')
    plt.xlabel('sample number')
    plt.ylabel('x error [m] and x variance')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(diff_from_gt[:, 1], label='error in y [m]')
    plt.plot(np.sqrt(total_sigma[:, 1, 1]), color='red', label='sigma y')
    plt.plot(-np.sqrt(total_sigma[:, 1, 1]), color='red')
    plt.xlabel('sample number')
    plt.ylabel('y error [m] and y variance')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(diff_theta, label='error in theta [m]')
    plt.plot(np.sqrt(total_sigma[:, 2, 2]), color='red', label='sigma theta')
    plt.plot(-np.sqrt(total_sigma[:, 2, 2]), color='red')
    plt.xlabel('sample number')
    plt.ylabel('theta error [m] and theta variance')
    plt.legend()
    plt.grid()

    plt.show(block=False)

    plt.figure()
    plt.imshow(sigma)
    plt.title('Covariance matrix')
    plt.show(block=False)


    # plot of landmard sigmot
    error_lm1 = landmarks[0,1:3] - total_mu[:,3:5]
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('Landmark 1 - error and std')
    plt.plot(error_lm1[:,0], label='error in x [m]')
    plt.plot(np.sqrt(total_sigma[:, 3,3 ]), color='red', label='sigma x')
    plt.plot(-np.sqrt(total_sigma[:, 3, 3]), color='red')
    plt.xlabel('sample number')
    plt.ylabel('x error [m] and x variance')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(error_lm1[:, 1], label='error in y [m]')
    plt.plot(np.sqrt(total_sigma[:, 4, 4]), color='red', label='sigma y')
    plt.plot(-np.sqrt(total_sigma[:, 4, 4]), color='red')
    plt.xlabel('sample number')
    plt.ylabel('y error [m] and y variance')
    plt.legend()
    plt.grid()

    plt.show(block=False)

    error_lm2 = landmarks[1, 1:3] - total_mu[:, 5:7]
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('Landmark 2 - error and std')
    plt.plot(error_lm2[:, 0], label='error in x [m]')
    plt.plot(np.sqrt(total_sigma[:, 5, 5]), color='red', label='sigma x')
    plt.plot(-np.sqrt(total_sigma[:, 5, 5]), color='red')
    plt.xlabel('sample number')
    plt.ylabel('x error [m] and x variance')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(error_lm2[:, 1], label='error in y [m]')
    plt.plot(np.sqrt(total_sigma[:, 6, 6]), color='red', label='sigma y')
    plt.plot(-np.sqrt(total_sigma[:, 6, 6]), color='red')
    plt.xlabel('sample number')
    plt.ylabel('y error [m] and y variance')
    plt.legend()
    plt.grid()

    plt.show(block=False)

    a = 3




def main():
    data_path = '/home/nadav/studies/mapping_and_perception_autonomous_robots/project_2/Ex3_verB'

    result_dir = r'/home/nadav/studies/mapping_and_perception_autonomous_robots/project_2/results'
    cur_date_time = time.strftime("%Y.%m.%d-%H.%M")

    result_dir_timed = os.path.join(result_dir, f'{cur_date_time}')
    print(f'saving to: {result_dir_timed}')
    os.makedirs(result_dir_timed, exist_ok=True)

    ekf_slam_func(data_path, result_dir_timed)
    if SAVE_VIDEO:
        create_video(result_dir_timed)


if __name__ == "__main__":
    np.random.seed(333)
    main()
