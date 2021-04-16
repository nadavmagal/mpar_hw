import numpy as np
import pykitti
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from common import load_data, exract_data_vectors

''' ========== Q2 - Extended kalman filter ========== '''
class ExtendedKalmanFilter:
    def __init__(self, meu_0, sigma_0):
        self.meu_t = meu_0
        self.sigma_t = sigma_0

        self.meu_t_predict = None
        self.sigma_t_predict = None

    def _filter_step(self, z_t, A_t, R_t, C_t, Q_t):
        self.meu_t_predict = A_t @ self.meu_t
        self.sigma_t_predict = A_t @ self.sigma_t @ A_t.T + R_t

        K_t = self.sigma_t_predict @ C_t.T @ np.linalg.inv(C_t @ self.sigma_t_predict @ C_t.T + Q_t)
        # K_t = np.eye(4)
        self.meu_t = self.meu_t_predict + K_t @ (z_t - C_t @ self.meu_t_predict)
        self.sigma_t = (np.eye(np.shape(K_t @ C_t)[0]) - K_t @ C_t) @ self.sigma_t_predict

        return self.meu_t, self.sigma_t

    def perform_ekf(self):
        return total_est_meu

def extended_kalman_filter(result_dir_timed, data):
    """ init data """
    car_w_coordinates_m, car_w_coordinates_lon_lat, car_yaw, car_vf, car_wz, delta_time, time_sec = exract_data_vectors(data)

    ''' plot GT LLA and ENU data '''
    plt.figure()
    plt.suptitle('GPS GT trajectory')
    plt.subplot(1, 2, 1)
    plt.scatter(car_w_coordinates_m[:, 0], car_w_coordinates_m[:, 1], color='blue', s=1, label='GT')
    start_index = 50
    plt.text(car_w_coordinates_m[start_index, 0], car_w_coordinates_m[start_index, 1], 'start')
    plt.text(car_w_coordinates_m[-start_index, 0], car_w_coordinates_m[-start_index, 1], 'end')
    plt.title('car coordinate ENU')
    plt.xlabel('east [m]')
    plt.ylabel('north [m]')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.scatter(car_w_coordinates_lon_lat[:, 0], car_w_coordinates_lon_lat[:, 1], color='blue', s=1, label='GT')
    plt.text(car_w_coordinates_lon_lat[start_index, 0], car_w_coordinates_lon_lat[start_index, 1], 'start')
    plt.text(car_w_coordinates_lon_lat[-start_index, 0], car_w_coordinates_lon_lat[-start_index, 1], 'end')
    plt.title('car coordinate LLA')
    plt.grid()
    plt.xlabel('lon [deg]')
    plt.ylabel('lat [deg]')
    plt.show(block=False)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time_sec, np.rad2deg(car_yaw))
    plt.grid()
    plt.xlabel('Time [sec]')
    plt.ylabel('yaw [deg]')

    plt.subplot(2,1,2)
    plt.plot(time_sec, car_vf)
    plt.grid()
    plt.xlabel('Time [sec]')
    plt.ylabel('vf [m/s]')

    plt.show(block=False)

    ''' add noise to GT data '''
    sigma_noise = 10
    noised_car_w_coordinates_m = car_w_coordinates_m + np.random.normal(0, sigma_noise, car_w_coordinates_m.shape)

    plt.figure()
    plt.scatter(car_w_coordinates_m[:, 0], car_w_coordinates_m[:, 1],s=1, color='blue', label='GT')
    plt.scatter(noised_car_w_coordinates_m[:, 0], noised_car_w_coordinates_m[:, 1], s=1, marker='x', color='red',
                label='noised_gt')
    plt.legend()
    plt.grid()
    plt.title('car coordinate ENU w/wo noise')
    plt.xlabel('east [m]')
    plt.ylabel('north [m]')
    plt.show(block=False)

    ''' Extended Kalman filter - constant velocity model '''
    # v_x_0 = 10
    # v_y_0 = 10
    theta_0 = car_yaw[0]
    sigma_0_x = 10
    sigma_0_y = 10
    sigma_0_yaw = 0.2

    sigma_a = 4

    meu_0 = np.array([car_w_coordinates_m[0][0], car_w_coordinates_m[0][1], theta_0])
    sigma_0 = np.diag([sigma_0_x ** 2, sigma_0_y ** 2, sigma_0_yaw ** 2])
    ekf = ExtendedKalmanFilter(meu_0, sigma_0)
    total_est_meu = ekf.perform_ekf()


    a=3

def main():
    basedir = '/home/nadav/studies/mapping_and_perception_autonomous_robots/kitti_data/orginaized_data'
    date = '2011_09_30'
    dataset_number = '0033'

    result_dir = r'/home/nadav/studies/mapping_and_perception_autonomous_robots/project_2/results'
    cur_date_time = time.strftime("%Y.%m.%d-%H.%M")

    result_dir_timed = os.path.join(result_dir, f'{cur_date_time}')
    print(f'saving to: {result_dir_timed}')
    os.makedirs(result_dir_timed, exist_ok=True)

    data = load_data(basedir, date, dataset_number)

    # Q2
    extended_kalman_filter(result_dir_timed, data)


if __name__ == "__main__":
    main()
