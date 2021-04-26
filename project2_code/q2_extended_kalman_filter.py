import numpy as np
import pykitti
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from frames_to_video import create_video
from common import load_data, exract_data_vectors, get_maxe_rmse, save_video_frame

''' ========== Q2 - Extended kalman filter ========== '''
SAVE_VIDEO = True


class ExtendedKalmanFilter:
    def __init__(self, meu_0, sigma_0):
        self.meu_t = meu_0
        self.sigma_t = sigma_0

        self.meu_t_predict = np.array([0., 0., 0.])
        self.sigma_t_predict = None

    def perform_ekf(self, data_vec, sigmot, save_video_dict):
        # TODO: https://github.com/motokimura/kalman_filter_witi_kitti
        sigma_x = sigmot['sigma_x']
        sigma_y = sigmot['sigma_y']
        sigma_vx = sigmot['sigma_vx']
        sigma_vy = sigmot['sigma_vy']
        sigma_yaw_rate = sigmot['sigma_yaw_rate']

        total_meu = []
        total_sigma = []
        dead_reckoning = []
        total_time_pass = 0.
        for ii in range(data_vec['car_w_coordinates_m'].shape[0]):
            if ii == 0:
                continue
            print(f'{ii}/{data_vec["car_w_coordinates_m"].shape[0]}')
            cur_car_coordinates = data_vec['car_w_coordinates_m'][ii]
            yaw = data_vec['yaw'][ii]
            vf = data_vec['vf'][ii]
            omega = data_vec['wz'][ii]
            dt = data_vec['delta_time'][ii]

            total_time_pass += dt

            ''' propagate '''
            meu_theta = self.meu_t[2]
            self.meu_t_predict[0] = self.meu_t[0] - (vf / omega) * np.sin(meu_theta) + (vf / omega) * np.sin(
                meu_theta + omega * dt)
            self.meu_t_predict[1] = self.meu_t[1] + (vf / omega) * np.cos(meu_theta) - (vf / omega) * np.cos(
                meu_theta + omega * dt)
            self.meu_t_predict[2] = self.meu_t[2] + (omega * dt)

            G = np.eye(3)
            G[0, 2] = -(vf / omega) * np.cos(meu_theta) + (vf / omega) * np.cos(meu_theta + omega * dt)
            G[1, 2] = -(vf / omega) * np.sin(meu_theta) + (vf / omega) * np.sin(meu_theta + omega * dt)

            if True:
                V = np.zeros([3, 2])
                V[0, 0] = -(1 / omega) * np.sin(meu_theta) + (1 / omega) * np.sin(meu_theta + omega * dt)
                V[0, 1] = (vf / (omega ** 2)) * np.sin(meu_theta) - (vf / (omega ** 2)) * np.sin(
                    meu_theta + omega * dt) + (
                                  vf / omega) * np.cos(meu_theta + omega * dt) * dt
                V[1, 0] = (1 / omega) * np.cos(meu_theta) - (1 / omega) * np.cos(meu_theta + omega * dt)
                V[1, 1] = -(vf / (omega ** 2)) * np.cos(meu_theta) + (vf / (omega ** 2)) * np.cos(
                    meu_theta + omega * dt) + (
                                  vf / omega) * np.sin(meu_theta + omega * dt) * dt
                V[2, 1] = dt
                R = np.diag([sigma_vx ** 2, sigma_yaw_rate ** 2])
                self.sigma_t_predict = G @ self.sigma_t @ G.T + V @ R @ V.T
            else:
                R = np.diag([sigma_vx ** 2, sigma_vy ** 2, sigma_yaw_rate ** 2])
                self.sigma_t_predict = G @ self.sigma_t @ G.T + R

            H = np.zeros([2, 3])
            H[0, 0] = 1
            H[1, 1] = 1

            Q = np.diag([sigma_x ** 2, sigma_y ** 2])

            ''' predict '''

            K = self.sigma_t_predict @ H.T @ np.linalg.inv(H @ self.sigma_t_predict @ H.T + Q)

            self.meu_t = self.meu_t_predict + K @ (cur_car_coordinates - H @ self.meu_t_predict)  # TODO: to be h
            self.sigma_t = (np.eye((K @ H).shape[0]) - K @ H) @ self.sigma_t_predict

            total_meu.append(self.meu_t.copy())
            total_sigma.append(self.sigma_t.copy())

            if total_time_pass > 5:
                if len(dead_reckoning) == 0:
                    cur_dead_reckoning = self.meu_t[0:2]
                else:
                    cur_dead_reckoning = cur_dead_reckoning + np.array([vf * np.cos(yaw), vf * np.sin(yaw)]) * dt

                dead_reckoning.append(cur_dead_reckoning)

            if SAVE_VIDEO:
                save_video_frame(save_video_dict['car_w_coordinates_m'][:ii,:], self.meu_t, self.sigma_t, dead_reckoning, ii,
                                 save_video_dict['noised_car_w_coordinates_m'][:ii,:], save_video_dict['result_dir_timed'],
                                 total_meu, total_time_pass, 'EKF')

        total_meu = np.array(total_meu)
        total_meu = np.vstack([np.array([0, 0, 0]), total_meu])
        total_sigma = np.array(total_sigma)
        return np.array(total_meu), total_sigma


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

    ''' Extended Kalman filter - constant velocity model - same initial condition as Kalman'''
    if False:
        sigmot = {
            'sigma_x': 3.,
            'sigma_y': 3.,
            'sigma_vx': 3.,
            'sigma_vy': 3.,
            'sigma_yaw_rate': 0.2,
            'sigma_yaw_initial': np.pi}

        meu_0 = np.array([noised_car_w_coordinates_m[0][0], noised_car_w_coordinates_m[0][1], car_yaw[0]])
        sigma_0 = np.diag([sigmot['sigma_x'] ** 2, sigmot['sigma_y'] ** 2, sigmot['sigma_yaw_initial'] ** 2])

        data_vec = {
            'car_w_coordinates_m': noised_car_w_coordinates_m,
            'yaw': car_yaw,
            'vf': car_vf,
            'wz': car_wz,
            'delta_time': delta_time}

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
        plt.title(
            fr'Extended Kalman - $\sigma_x=\sigma_y={sigmot["sigma_x"]}, \sigma_v={sigmot["sigma_vx"]}, \sigma_\omega={sigmot["sigma_yaw_rate"]}$ - RMSE={round(rmse, 2)} [m], maxE={round(max_E, 2)} [m]')
        plt.xlabel('east [m]')
        plt.ylabel('north [m]')
        plt.show(block=False)

        a = 3

    ''' add noise to IMU '''
    wz_sigma_noise = 0.2
    noised_wz = car_wz + np.random.normal(0, wz_sigma_noise, car_wz.shape)

    vf_sigma_noise = 2
    noised_car_vf = car_vf + np.random.normal(0, vf_sigma_noise, car_vf.shape)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.scatter(time_sec, car_wz, s=1, label='gt yaw rate')
    plt.scatter(time_sec, noised_wz, s=1, label='noised yaw rate')
    plt.grid()
    plt.xlabel('Time [sec]')
    plt.ylabel('yaw rate [deg/sec]')

    plt.subplot(2, 1, 2)
    plt.scatter(time_sec, car_vf, s=1, label='gt vf')
    plt.scatter(time_sec, noised_car_vf, s=1, label='noised vf')
    plt.grid()
    plt.xlabel('Time [sec]')
    plt.ylabel('vf [m/s]')

    plt.show(block=False)

    ''' EKF - with suitable initial condition '''
    sigmot = {
        'sigma_x': 3.,
        'sigma_y': 3.,
        'sigma_vx': 3.,
        'sigma_vy': 3.,
        'sigma_yaw_rate': 0.3,
        'sigma_yaw_initial': np.pi}

    meu_0 = np.array([noised_car_w_coordinates_m[0][0], noised_car_w_coordinates_m[0][1], car_yaw[0]])
    sigma_0 = np.diag([sigmot['sigma_x'] ** 2, sigmot['sigma_y'] ** 2, sigmot['sigma_yaw_initial'] ** 2])

    data_vec = {
        'car_w_coordinates_m': noised_car_w_coordinates_m,
        'yaw': car_yaw,
        'vf': noised_car_vf,
        'wz': noised_wz,
        'delta_time': delta_time}

    save_video_dict = {'car_w_coordinates_m': car_w_coordinates_m,
                       'noised_car_w_coordinates_m': noised_car_w_coordinates_m,
                       'result_dir_timed': result_dir_timed}

    ekf = ExtendedKalmanFilter(meu_0, sigma_0)
    total_est_meu, total_sigma = ekf.perform_ekf(data_vec, sigmot, save_video_dict)
    if SAVE_VIDEO:
        create_video(result_dir_timed)

    max_E, rmse = get_maxe_rmse(car_w_coordinates_m, total_est_meu)
    plt.figure()
    plt.scatter(car_w_coordinates_m[:, 0], car_w_coordinates_m[:, 1], s=1, color='blue', label='GT')
    plt.scatter(noised_car_w_coordinates_m[:, 0], noised_car_w_coordinates_m[:, 1], s=1, marker='x', color='red',
                label='noised_gt')
    plt.scatter(total_est_meu[:, 0], total_est_meu[:, 1], s=1, marker='x', color='green', label='kalman CV')
    plt.legend()
    plt.grid()
    plt.title(
        fr'Extended Kalman - $\sigma_x=\sigma_y={sigma_noise}, \sigma_v={vf_sigma_noise}, \sigma_\omega={wz_sigma_noise}$ - RMSE={round(rmse, 2)} [m], maxE={round(max_E, 2)} [m]')
    plt.xlabel('east [m]')
    plt.ylabel('north [m]')
    plt.show(block=False)
    a = 3

    diff_x = car_w_coordinates_m[:, 0] - total_est_meu[:, 0]
    diff_y = car_w_coordinates_m[:, 1] - total_est_meu[:, 1]
    total_est_theta = total_est_meu[:, 2]
    mod_car_yaw = [cur_yaw if ii < 975 else cur_yaw - 2*np.pi for ii, cur_yaw in enumerate(car_yaw)] # np.where(car_yaw<2, car_yaw, car_yaw-2*np.pi)
    diff_theta = mod_car_yaw - total_est_theta
    # diff_theta = np.where(diff_theta<np.pi/2, diff_theta, diff_theta-np.pi)
    # diff_theta = np.deg2rad(np.mod(np.rad2deg(car_yaw) - np.rad2deg(total_est_meu[:, 2]),180))

    # plt.figure()
    # plt.plot(mod_car_yaw)
    # plt.plot(total_est_theta)
    # plt.plot(diff_theta)
    # plt.show(block=False)

    plt.figure(figsize=[15, 10])
    plt.subplot(3, 1, 1)
    plt.plot(diff_x, label='error in x [m]')
    plt.plot(np.sqrt(total_sigma[:, 0, 0]), color='red', label='sigma x')
    plt.plot(-np.sqrt(total_sigma[:, 0, 0]), color='red')
    # plt.title('Error in x [m] and variance x')
    plt.xlabel('sample number')
    plt.ylabel('x error [m] and x variance')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(diff_y, label='error in y [m]')
    plt.plot(np.sqrt(total_sigma[:, 1, 1]), color='red', label='sigma y')
    plt.plot(-np.sqrt(total_sigma[:, 1, 1]), color='red')
    # plt.title('Error in y [m] and variance y')
    plt.xlabel('sample number')
    plt.ylabel('y error [m] and y variance')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(diff_theta, label=r'error in $\theta$ [rad]')
    plt.plot(np.sqrt(total_sigma[:, 2, 2]), color='red', label=r'sigma $\theta$')
    plt.plot(-np.sqrt(total_sigma[:, 2, 2]), color='red')
    # plt.title(r'Error in $\theta$ [rad] and variance $\theta$')
    plt.xlabel('sample number')
    plt.ylabel(r'$\theta$ error [rad] and $\theta$ variance')
    plt.grid()
    plt.legend()

    plt.show(block=False)
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
    np.random.seed(0)
    main()
