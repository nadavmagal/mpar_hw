import numpy as np
import pykitti
import os
import time
import matplotlib.pyplot as plt
from frames_to_video import create_video
import pandas as pd
from common import load_data, exract_data_vectors, get_maxe_rmse, save_video_frame

SAVE_VIDEO = True


''' ========== Q1 - kalman filter ========== '''


class KalmanFilter:

    def __init__(self, meu_0, sigma_0):
        self.meu_t = meu_0
        self.sigma_t = sigma_0

        self.meu_t_predict = None
        self.sigma_t_predict = None

    def filter_step(self, u_t, z_t, A_t, B_t, R_t, C_t, Q_t):
        self.meu_t_predict = A_t @ self.meu_t + B_t @ u_t
        self.sigma_t_predict = A_t @ self.sigma_t @ A_t.T + R_t

        K_t = self.sigma_t_predict @ C_t.T @ np.linalg.inv(C_t @ self.sigma_t_predict @ C_t.T + Q_t)
        self.meu_t = self.meu_t_predict + K_t @ (z_t - C_t @ self.meu_t_predict)
        self.sigma_t = (np.ones_like(K_t @ C_t) - K_t @ C_t) @ self.sigma_t_predict


class KalmanFilterConstantVelocity:
    def __init__(self, meu_0, sigma_0):
        self.meu_t = meu_0
        self.sigma_t = sigma_0

        self.meu_t_predict = None
        self.sigma_t_predict = None

    def filter_step(self, z_t, A_t, R_t, C_t, Q_t):
        self.meu_t_predict = A_t @ self.meu_t
        self.sigma_t_predict = A_t @ self.sigma_t @ A_t.T + R_t

        K_t = self.sigma_t_predict @ C_t.T @ np.linalg.inv(C_t @ self.sigma_t_predict @ C_t.T + Q_t)
        # K_t = np.eye(4)
        self.meu_t = self.meu_t_predict + K_t @ (z_t - C_t @ self.meu_t_predict)
        self.sigma_t = (np.eye(np.shape(K_t @ C_t)[0]) - K_t @ C_t) @ self.sigma_t_predict

        return self.meu_t, self.sigma_t


def kalman_filter(result_dir_timed, data):
    """ init data """
    car_w_coordinates_m, car_w_coordinates_lon_lat, _, _, _, delta_time, time_sec = exract_data_vectors(data)

    delta_x = car_w_coordinates_m[:, 0][1::] - car_w_coordinates_m[:, 0][:-1:]
    delta_y = car_w_coordinates_m[:, 1][1::] - car_w_coordinates_m[:, 1][:-1:]

    vx = np.divide(delta_x, delta_time[1::])
    vy = np.divide(delta_y, delta_time[1::])
    vxvy = np.array([vx, vy]).T

    vxvy = np.vstack([np.array([0, 0]), vxvy])


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

    ''' Kalman filter - constant velocity model '''

    # initialization
    x_0 = 0.
    y_0 = 0.
    v_x_0 = 0.
    v_y_0 = 0.
    sigma_0_x = 3.
    sigma_0_y = 3.
    sigma_0_vx = 3.
    sigma_0_vy = 3.
    sigma_a = 1.5
    total_est_meu, dead_reckoning, total_est_sigma = kalman_constant_velocity(delta_time, noised_car_w_coordinates_m,
                                                                              sigma_0_vx, sigma_0_vy, sigma_0_x,
                                                                              sigma_0_y, sigma_a, x_0, y_0, v_x_0,
                                                                              v_y_0, result_dir_timed,
                                                                              car_w_coordinates_m, vxvy)

    if SAVE_VIDEO:
        create_video(result_dir_timed)
    # total_est_meu = kalman_constant_velocity_flip(delta_time, noised_car_w_coordinates_m, sigma_0_vx, sigma_0_vy,
    #                                               sigma_0_x,
    #                                               sigma_0_y, sigma_a, x_0, y_0, v_x_0, v_y_0)
    max_E, rmse = get_maxe_rmse(car_w_coordinates_m, total_est_meu)
    print(f'kalman rmse={round(rmse, 3)} :: maxE={round(max_E, 3)}')

    max_E_dead, rmse_dead = get_maxe_rmse(car_w_coordinates_m[-dead_reckoning.shape[0]:, :], dead_reckoning)
    print(f'dead_reckoning rmse={round(rmse_dead, 3)} :: maxE={round(max_E_dead, 3)}')

    # rmse, total_est_meu = explore_kalman_cv(car_w_coordinates_m, delta_time, noised_car_w_coordinates_m)

    plt.figure()
    plt.title(rf'Kalman constant velocity $\sigma_x = \sigma_y={sigma_noise}$ - RMSE={round(rmse, 2)} [m], maxE={round(max_E, 2)} [m]')
    plt.scatter(car_w_coordinates_m[:, 0], car_w_coordinates_m[:, 1], s=1, label='GT')
    plt.scatter(noised_car_w_coordinates_m[:, 0], noised_car_w_coordinates_m[:, 1], s=1, marker='x', color='red',
                label='noised_gt')
    plt.scatter(total_est_meu[:, 0], total_est_meu[:, 1], s=1, marker='x', color='green', label='kalman CV')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.legend()
    plt.show(block=False)
    a = 3

    diff_x = car_w_coordinates_m[:, 0] - total_est_meu[:, 0]
    diff_y = car_w_coordinates_m[:, 1] - total_est_meu[:, 1]
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(diff_x, label='error in x [m]')
    plt.plot(np.sqrt(total_est_sigma[:, 0, 0]), color='red', label='sigma x')
    plt.plot(-np.sqrt(total_est_sigma[:, 0, 0]), color='red', label='sigma x')
    plt.title('Error in x [m] and variance x')
    plt.xlabel('sample number')
    plt.ylabel('x error [m] and x variance')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(diff_y, label='error in y [m]')
    plt.plot(np.sqrt(total_est_sigma[:, 1, 1]), color='red', label='sigma y')
    plt.plot(-np.sqrt(total_est_sigma[:, 1, 1]), color='red', label='sigma y')
    plt.title('Error in y [m] and variance y')
    plt.xlabel('sample number')
    plt.ylabel('y error [m] and y variance')
    plt.grid()

    plt.show(block=False)
    a = 3


def explore_kalman_cv(car_w_coordinates_m, delta_time, noised_car_w_coordinates_m):
    total_results = []
    jj = 0
    for cur_v0 in range(0, 10):
        for cur_sigma_0_v in [0.1, 0.5, 1, 2, 4, 8, 10, 12, 15, 20]:
            for cur_sigma_a in [0.1, 0.5, 1, 2, 4, 8, 10, 12, 15, 20]:
                print(f'{jj}/{10 * 10 * 10}')
                jj += 1
                v_x_0 = cur_v0
                v_y_0 = cur_v0
                sigma_0_x = 10
                sigma_0_y = 10
                sigma_0_vx = cur_sigma_0_v
                sigma_0_vy = cur_sigma_0_v
                sigma_a = cur_sigma_a
                try:
                    total_est_meu, _ = kalman_constant_velocity(delta_time, noised_car_w_coordinates_m, sigma_0_vx,
                                                                sigma_0_vy, sigma_0_x,
                                                                sigma_0_y, sigma_a, v_x_0, v_y_0)
                except:
                    print('fail')
                    continue

                ex_ey = car_w_coordinates_m[600:, :] - total_est_meu[600:, 0:2]
                max_E = np.max(np.sum(np.abs(ex_ey[100:, :]), axis=1))
                rmse = np.sqrt(np.sum(np.power(ex_ey, 2)) / total_est_meu.shape[0])
                total_results.append(np.array([cur_v0, cur_sigma_0_v, cur_sigma_a, rmse, max_E]))
    np.save('total_results.npy', total_results)
    total_results = np.array(total_results)
    cur_v0 = total_results[:, 0]
    cur_sigma_0_v = total_results[:, 1]
    cur_sigma_a = total_results[:, 2]
    rmse = total_results[:, 3]
    max_E = total_results[:, 4]
    plt.figure()
    cur_x = range(len(cur_v0))
    plt.plot()
    plt.subplot(2, 1, 1)
    plt.plot(cur_x, rmse)
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(cur_x, max_E)
    plt.grid()
    return rmse, total_est_meu


def kalman_constant_velocity(delta_time, noised_car_w_coordinates_m, sigma_0_vx, sigma_0_vy, sigma_0_x, sigma_0_y,
                             sigma_a, x_0, y_0, v_x_0, v_y_0, result_dir_timed, car_w_coordinates_m, vxvy):
    total_est_meu = []
    total_est_sigma = []
    dead_reckoning = []
    total_time_pass = 0.
    for ii, cur_car_coord, cur_delta_t, cur_vxvy in zip(range(noised_car_w_coordinates_m.shape[0]), noised_car_w_coordinates_m,
                                              delta_time, vxvy):
        print(f'{ii}/{noised_car_w_coordinates_m.shape[0]}')
        # cur_delta_t = 0.1*1e3
        if ii == 0:
            meu_0 = np.array([x_0, y_0, v_x_0, v_y_0])
            sigma_0 = np.diag([sigma_0_x ** 2, sigma_0_y ** 2, (sigma_0_vx) ** 2, (sigma_0_vy) ** 2])
            filter_cv = KalmanFilterConstantVelocity(meu_0, sigma_0)
            continue
        total_time_pass += cur_delta_t
        z_t = cur_car_coord

        A_t = np.array([[1, 0, cur_delta_t, 0],
                        [0, 1, 0, cur_delta_t],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        R_t = np.zeros([4, 4])
        R_t[2, 2] = cur_delta_t * sigma_a ** 2
        R_t[3, 3] = cur_delta_t * sigma_a ** 2

        C_t = np.zeros([2, 4])
        C_t[0, 0] = 1
        C_t[1, 1] = 1

        Q_t = np.diag([(sigma_0_vx) ** 2, (sigma_0_vy) ** 2])  # TODO: maybe change this not to be like sigma 0

        cur_est_meu_t, cur_est_sigma_t = filter_cv.filter_step(z_t, A_t, R_t, C_t, Q_t)
        total_est_meu.append(cur_est_meu_t)
        total_est_sigma.append(cur_est_sigma_t)

        if total_time_pass > 5:
            if len(dead_reckoning) == 0:
                cur_dead_reckoning = cur_est_meu_t[0:2]
            else:
                cur_dead_reckoning = cur_dead_reckoning + cur_vxvy * cur_delta_t

            dead_reckoning.append(cur_dead_reckoning)

        if SAVE_VIDEO:
            save_video_frame(car_w_coordinates_m[:ii,:], cur_est_meu_t, cur_est_sigma_t, dead_reckoning, ii,
                             noised_car_w_coordinates_m[:ii,:], result_dir_timed, total_est_meu, total_time_pass, 'Kalman')

    total_est_meu = np.array(total_est_meu)
    total_est_meu = np.vstack([np.array([0, 0, 0, 0]), total_est_meu])
    dead_reckoning = np.array(dead_reckoning)
    total_est_sigma = np.array(total_est_sigma)
    return total_est_meu, dead_reckoning, total_est_sigma


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

    # Q1
    kalman_filter(result_dir_timed, data)
    a = 3


if __name__ == "__main__":
    # np.random.seed(0)
    main()
