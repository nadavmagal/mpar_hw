import pykitti
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse

vid_fig, vid_axs = plt.subplots(1, 2, figsize=[15, 10], dpi=300)


def load_data(basedir, date, dataset_number):
    data = pykitti.raw(basedir, date, dataset_number)
    return data


def exract_data_vectors(data):
    oxts = data.oxts
    delta_time = np.append([0], np.array(
        [cur.total_seconds() for cur in np.array(data.timestamps[1::]) - np.array(data.timestamps[:-1:])]))
    time_sec = np.cumsum(delta_time)

    point_imu = np.array([0, 0, 0, 1])
    car_w_coordinates_m = []
    car_w_coordinates_lon_lat = []
    car_yaw = []
    car_vf = []
    car_wz = []
    for cur_oxt in oxts:
        car_w_coordinates_m.append(cur_oxt.T_w_imu.dot(point_imu)[0:2])
        car_w_coordinates_lon_lat.append([cur_oxt.packet.lon, cur_oxt.packet.lat])
        car_yaw.append(cur_oxt.packet.yaw)
        car_vf.append(cur_oxt.packet.vf)
        car_wz.append(cur_oxt.packet.wz)
    car_w_coordinates_m = np.array(car_w_coordinates_m)
    car_w_coordinates_lon_lat = np.array(car_w_coordinates_lon_lat)
    car_yaw = np.array(car_yaw)
    car_vf = np.array(car_vf)
    car_wz = np.array(car_wz)
    return car_w_coordinates_m, car_w_coordinates_lon_lat, car_yaw, car_vf, car_wz, delta_time, time_sec


def get_maxe_rmse(car_w_coordinates_m, total_est_meu):
    ex_ey = car_w_coordinates_m[100:, 0:2] - total_est_meu[100:, 0:2]
    max_E = np.max(np.sum(np.abs(ex_ey[100:, :]), axis=1))
    rmse = np.sqrt(np.sum(np.power(ex_ey, 2)) / total_est_meu.shape[0])
    return max_E, rmse


def save_video_frame(car_w_coordinates_m, cur_est_meu_t, cur_est_sigma_t, dead_reckoning, ii,
                     noised_car_w_coordinates_m, result_dir_timed, total_est_meu, total_time_pass, method):
    plt.figure(vid_fig)
    ax0 = vid_axs[0]
    ax0.clear()
    ax0.scatter(car_w_coordinates_m[:, 0], car_w_coordinates_m[:, 1], s=1, color='blue', label='GT')
    ax0.scatter(noised_car_w_coordinates_m[:, 0], noised_car_w_coordinates_m[:, 1], s=1, marker='x',
                color='red',
                label='noised_gt')
    ax0.scatter(np.array(total_est_meu)[:, 0], np.array(total_est_meu)[:, 1], color='green', marker='x', s=1,
                label='Kalman - CV')
    if total_time_pass > 5:
        if method=='Kalman':
            cur_ellipse = Ellipse((cur_est_meu_t[0], cur_est_meu_t[1]), cur_est_sigma_t[0][0],
                                  cur_est_sigma_t[1][1],
                                  np.rad2deg(np.arctan2(cur_est_meu_t[3], cur_est_meu_t[2])), edgecolor='green',
                                  fc='None', lw=2)
        elif method == 'EKF':
            cur_ellipse = Ellipse((cur_est_meu_t[0], cur_est_meu_t[1]), 4 * cur_est_sigma_t[0][0],
                                  4 * cur_est_sigma_t[1][1],
                                  np.rad2deg(cur_est_meu_t[2]), edgecolor='green',
                                  fc='None', lw=2)
        ax0.add_patch(cur_ellipse)
        ax0.scatter(np.array(dead_reckoning)[:, 0], np.array(dead_reckoning)[:, 1], color='magenta', marker='x',
                    s=1, label='dead reckoning')
    ax0.legend()
    ax0.grid()
    ax0.set_title(f'{method} - car trajectory - time={round(total_time_pass, 2)} sec')
    ax0.set_xlabel('east [m]')
    ax0.set_ylabel('north [m]')
    ax1 = vid_axs[1]
    ax1.clear()
    ax1.imshow(cur_est_sigma_t)
    ax1.set_title('state covariance matrix')
    image_path = os.path.join(result_dir_timed, f'{ii}.png')
    vid_fig.savefig(image_path, dpi=150)
