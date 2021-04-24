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
from q3_efk_slam.tools import *
from q3_efk_slam.prediction_step import prediction_step

def ekf_slam_func(data_path):

    # % Read world data, i.e. landmarks. The true landmark positions are not given to the robot
    landmarks = read_world(os.path.join(data_path, 'world.dat'))

    # % load landmarks;
    # % Read sensor readings, i.e. odometry and range-bearing sensor
    data = read_data(os.path.join(data_path, 'sensor_data.dat'))
    # TODO: do we have the delta r1 r2 t or we have the absolute values
    # %load data;

    INF = 1000
    # % Get the number of landmarks in the map
    N = landmarks.shape[0]

    '''
    # % observedLandmarks is a vector that keeps track of which landmarks have been observed so far.
    # % observedLandmarks(i) will be true if the landmark with id = i has been observed at some point by the robot
    # observedLandmarks = repmat(false,1,N);
    observedLandmarks = np.array([False] * N)
    '''

    # % TODO Initialize belief:
    '''
    # % mu: 2N+3x1 vector representing the mean of the normal distribution
    # % The first 3 components of mu correspond to the pose of the robot,
    # % and the landmark poses (xi, yi) are stacked in ascending id order.
    # % sigma: (2N+3)x(2N+3) covariance matrix of the normal distribution
    '''
    x0, y0, theta0 = 0., 0., 0.
    mu = np.hstack([np.array([x0, y0, theta0]), np.zeros(2*N)])

    sigma_x0, sigma_y0, sigma_theta0 = 1., 1., 1.
    sigma_xx_0 = np.diag([sigma_x0 ** 2, sigma_y0 ** 2, sigma_theta0 ** 2])
    sigma_mm_0 = np.diag([INF]*2*N)
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
    for t in range(len(data['timestep'])):
        # % Perform the prediction step of the EKF
        a=3

        mu, sigma = prediction_step(mu, sigma, data['timestep'][t]['odometry'])
        '''
        % Perform the correction step of the EKF
       [mu, sigma, observedLandmarks] = correction_step(mu, sigma, data.timestep(t).sensor, observedLandmarks);

        %Generate visualization plots of the current state of the filter
        plot_state(mu, sigma, landmarks, t, observedLandmarks, data.timestep(t).sensor, showGui);
        disp('Current state vector:')
        disp('mu = '), disp(mu)
    end

    disp('Final system covariance matrix:'), disp(sigma)
    % Display the final state estimate
    disp('Final robot pose:')
    disp('mu_robot = '), disp(mu(1:3)), disp('sigma_robot = '), disp(sigma(1:3,1:3))
    '''

def main():
    data_path = '/home/nadav/studies/mapping_and_perception_autonomous_robots/project_2/Ex3_verB'
    ekf_slam_func(data_path)

if __name__ == "__main__":
    main()



