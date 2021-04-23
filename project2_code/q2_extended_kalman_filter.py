import numpy as np
import pykitti
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from common import load_data, exract_data_vectors, get_maxe_rmse

''' ========== Q2 - Extended kalman filter ========== '''


class ExtendedKalmanFilter:
    def __init__(self, meu_0, sigma_0):
        self.meu_t = meu_0
        self.sigma_t = sigma_0

        self.meu_t_predict = np.array([0,0,0])
        self.sigma_t_predict = None

    # def _filter_step(self, z_t, A_t, R_t, C_t, Q_t):
    #     self.meu_t_predict = A_t @ self.meu_t
    #     self.sigma_t_predict = A_t @ self.sigma_t @ A_t.T + R_t
    #
    #     K_t = self.sigma_t_predict @ C_t.T @ np.linalg.inv(C_t @ self.sigma_t_predict @ C_t.T + Q_t)
    #     # K_t = np.eye(4)
    #     self.meu_t = self.meu_t_predict + K_t @ (z_t - C_t @ self.meu_t_predict)
    #     self.sigma_t = (np.eye(np.shape(K_t @ C_t)[0]) - K_t @ C_t) @ self.sigma_t_predict
    #
    #     return self.meu_t, self.sigma_t

    def perform_ekf(self, data_vec, sigmot):
        # TODO: https://github.com/motokimura/kalman_filter_witi_kitti
        sigma_x = sigmot['sigma_x']
        sigma_y = sigmot['sigma_y']
        sigma_vx = sigmot['sigma_vx']
        sigma_vy = sigmot['sigma_vy']
        sigma_yaw = sigmot['sigma_yaw']

        total_meu = []
        total_sigma = []

        for ii in range(data_vec['noised_car_w_coordinates_m'].shape[0]):
            if ii == 0:
                continue
            cur_car_coordinates = data_vec['noised_car_w_coordinates_m'][ii]
            theta = data_vec['yaw'][ii]
            vf = data_vec['vf'][ii]
            omega = data_vec['wz'][ii]
            dt = data_vec['delta_time'][ii]

            self.meu_t_predict[0] = self.meu_t[0] - (vf / omega) * np.sin(theta) + (vf / omega) * np.sin(theta + omega * dt)
            self.meu_t_predict[1] = self.meu_t[1] + (vf / omega) * np.cos(theta) - (vf / omega) * np.cos(theta + omega * dt)
            self.meu_t_predict[2] = self.meu_t[2] + (omega * dt)

            G = np.eye(3)
            G[0, 2] = -(vf / omega) * np.cos(theta) + (vf / omega) * np.cos(theta + omega * dt)
            G[1, 2] = -(vf / omega) * np.sin(theta) + (vf / omega) * np.sin(theta + omega * dt)

            # V = np.zeros([3, 2])
            # V[0, 0] = -(1 / omega) * np.sin(theta) + (1 / omega) * np.sin(theta + omega * dt)
            # V[0, 1] = (vf / (omega ** 2)) * np.sin(theta) - (vf / (omega ** 2)) * np.sin(theta + omega * dt) + (
            #         vf / omega) * np.cos(theta + omega * dt) * dt
            # V[1, 0] = (1 / omega) * np.cos(theta) - (1 / omega) * np.cos(theta + omega * dt)
            # V[1, 1] = -(vf / (omega ** 2)) * np.cos(theta) + (vf / (omega ** 2)) * np.cos(theta + omega * dt) - (
            #         vf / omega) * np.sin(theta + omega * dt) * dt
            # V[2, 1] = dt

            R = np.diag([sigma_vx ** 2, sigma_vy ** 2, sigma_yaw ** 2])
            R
            # R = np.diag([sigma_vx ** 2, sigma_yaw ** 2])

            self.sigma_t_predict = G @ self.sigma_t @ G.T + R
            # self.sigma_t_predict = G @ self.sigma_t @ G.T + V @ R @ V.T

            H = np.zeros([2, 3])
            H[0, 0] = 1
            H[1, 1] = 1

            Q = np.diag([sigma_x ** 2, sigma_y ** 2])

            K = self.sigma_t_predict @ H.T @ np.linalg.inv(H @ self.sigma_t_predict @ H.T + Q)

            self.meu_t = self.meu_t_predict + K @ (
                    cur_car_coordinates - H @ self.meu_t_predict)  # TODO: suppose to be h
            self.sigma_t = (np.eye((K @ H).shape[0]) - K @ H) @ self.sigma_t_predict

            total_meu.append(self.meu_t.copy())
            total_sigma.append(self.sigma_t.copy())
        total_meu = np.array(total_meu)
        total_meu = np.vstack([np.array([0, 0, 0]), total_meu])
        return np.array(total_meu)


def extended_kalman_filter(result_dir_timed, data):
    """ init data """
    car_w_coordinates_m, car_w_coordinates_lon_lat, car_yaw, car_vf, car_wz, delta_time, time_sec = exract_data_vectors(
        data)

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
    plt.subplot(3, 1, 1)
    plt.plot(time_sec, np.rad2deg(car_yaw))
    plt.grid()
    plt.xlabel('Time [sec]')
    plt.ylabel('yaw [deg]')

    plt.subplot(3, 1, 2)
    plt.plot(time_sec, car_vf)
    plt.grid()
    plt.xlabel('Time [sec]')
    plt.ylabel('vf [m/s]')

    plt.subplot(3, 1, 3)
    plt.plot(time_sec, car_wz)
    plt.grid()
    plt.xlabel('Time [sec]')
    plt.ylabel('angular rate [rad/s]')

    plt.show(block=False)

    ''' add noise to GT data '''
    sigma_noise = 3
    noised_car_w_coordinates_m = car_w_coordinates_m + np.random.normal(0, sigma_noise, car_w_coordinates_m.shape)

    plt.figure()
    plt.scatter(car_w_coordinates_m[:, 0], car_w_coordinates_m[:, 1], s=1, color='blue', label='GT')
    plt.scatter(noised_car_w_coordinates_m[:, 0], noised_car_w_coordinates_m[:, 1], s=1, marker='x', color='red',
                label='noised_gt')
    plt.legend()
    plt.grid()
    plt.title('car coordinate ENU w/wo noise')
    plt.xlabel('east [m]')
    plt.ylabel('north [m]')
    plt.show(block=False)

    ''' Extended Kalman filter - constant velocity model '''

    sigma_x = 3
    sigma_y = 3
    sigma_0_yaw = 0.2

    sigma_vx = 2  # TODO: change
    sigma_vy = 2  # TODO: change

    meu_0 = np.array([noised_car_w_coordinates_m[0][0], noised_car_w_coordinates_m[0][1], car_yaw[0]])
    sigma_0 = np.diag([sigma_x ** 2, sigma_y ** 2, sigma_0_yaw ** 2])

    data_vec = {
        'noised_car_w_coordinates_m': noised_car_w_coordinates_m,
        'yaw': car_yaw,
        'vf': car_vf,
        'wz': car_wz,
        'delta_time': delta_time}
    sigmot = {
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'sigma_vx': sigma_vx,
        'sigma_vy': sigma_vy,
        'sigma_yaw': sigma_0_yaw}
    ekf = ExtendedKalmanFilter(meu_0, sigma_0)
    total_est_meu = ekf.perform_ekf(data_vec, sigmot)

    max_E, rmse = get_maxe_rmse(car_w_coordinates_m, total_est_meu)

    plt.figure()
    plt.scatter(car_w_coordinates_m[:, 0], car_w_coordinates_m[:, 1], s=1, color='blue', label='GT')
    plt.scatter(noised_car_w_coordinates_m[:, 0], noised_car_w_coordinates_m[:, 1], s=1, marker='x', color='red',
                label='noised_gt')
    plt.scatter(total_est_meu[:, 0], total_est_meu[:, 1], s=1, marker='x', color='green', label='kalman CV')
    plt.legend()
    plt.grid()
    plt.title(f'Extended Kalman - RMSE={round(rmse, 2)} [m], maxE={round(max_E, 2)} [m]')
    plt.xlabel('east [m]')
    plt.ylabel('north [m]')
    plt.show(block=False)

    a = 3


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
    np.random.seed(0)
    main()
