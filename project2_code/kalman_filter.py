import numpy as np
import pykitti
import os
import time
import matplotlib.pyplot as plt

class KALMAM_FILTER():
    # https: // arxiv.org / pdf / 1204.0375.pdf
    #TODO - remove
    def __init__(self, meu_0, sigma_0):
        self.meu_t = meu_0
        self.sigma_t = sigma_0

        self.meu_t_predict = None
        self.sigma_t_predict = None

        self.A = None
        self.B = None
        self.C = None
        self.Q = None
        self.K = None

    def predict(self, ):
        self.meu_t_predict = self.A @ self.meu_t

def load_data(basedir, date, dataset_number):
    data = pykitti.raw(basedir, date, dataset_number)
    return data


def hw_2(result_dir_timed, data, sigma, skip_frames=1):
    '''' init data '''
    oxts = data.oxts[::skip_frames]
    # cam2 = list(data.cam2)[::skip_frames]
    # velo = list(data.velo)[::skip_frames]
    T_velo_imu = data.calib.T_velo_imu
    time_sec = np.append([0], np.cumsum(np.array(
        [cur.microseconds * 1e-6 for cur in np.array(data.timestamps[1::]) - np.array(data.timestamps[:-1:])])))

    ''' init figures '''
    gt_noised_figure = plt.figure()

    ''' data calculation '''
    point_imu = np.array([0, 0, 0, 1])
    car_w_coordinates_m = np.array([o.T_w_imu.dot(point_imu) for o in oxts])[:, 0:2]
    car_w_coordinates_lon_lat = np.array([[o.packet.lon, o.packet.lat] for o in oxts])

    ''''''
    plot_gt_trajectory(car_w_coordinates_lon_lat, car_w_coordinates_m, gt_noised_figure, result_dir_timed)

    ''' add noise to GT data '''
    noised_car_w_coordinates_m = car_w_coordinates_m + np.random.normal(0, sigma, car_w_coordinates_m.shape)

    plt.figure(gt_noised_figure)
    plt.subplot(1,2,1)
    plt.scatter(noised_car_w_coordinates_m[:,0], noised_car_w_coordinates_m[:,1],s=1, marker='x', color='red', label='noised_gt')
    plt.legend()
    plt.show(block=False)

    a = 3


def plot_gt_trajectory(car_w_coordinates_lon_lat, car_w_coordinates_m, gt_noised_figure, result_dir_timed):
    ''' plotting GPS GT trajectory '''
    # plt.figure(figsize=[15, 10], dpi=500)
    plt.figure(gt_noised_figure)
    plt.suptitle('GPS GT trajectory')
    plt.subplot(1, 2, 1)
    plt.scatter(car_w_coordinates_m[:, 0], car_w_coordinates_m[:, 1], c=range(len(car_w_coordinates_m[:, 0])),
                cmap='cool', label='GT')
    start_index = 50
    plt.text(car_w_coordinates_m[start_index, 0], car_w_coordinates_m[start_index, 1], 'start')
    plt.text(car_w_coordinates_m[-start_index, 0], car_w_coordinates_m[-start_index, 1], 'end')
    plt.title('car coordinate ENU')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(car_w_coordinates_lon_lat[:, 0], car_w_coordinates_lon_lat[:, 1],
                c=range(len(car_w_coordinates_lon_lat[:, 0])), cmap='cool', label='GT')
    plt.text(car_w_coordinates_lon_lat[start_index, 0], car_w_coordinates_lon_lat[start_index, 1], 'start')
    plt.text(car_w_coordinates_lon_lat[-start_index, 0], car_w_coordinates_lon_lat[-start_index, 1], 'end')
    plt.title('car coordinate LLA')
    plt.grid()
    plt.xlabel('lon [deg]')
    plt.ylabel('lat [deg]')
    plt.legend()
    plt.show(block=False)
    # plt.savefig(os.path.join(result_dir_timed, f'GPS_GT_trajectory.png'))


if __name__ == "__main__":
    basedir = '/home/nadav/studies/mapping_and_perception_autonomous_robots/kitti_data/orginaized_data'
    date = '2011_09_30'
    dataset_number = '0033'

    sigma = 10

    result_dir = r'/home/nadav/studies/mapping_and_perception_autonomous_robots/project_2/results'
    cur_date_time = time.strftime("%Y.%m.%d-%H.%M")

    result_dir_timed = os.path.join(result_dir, f'{cur_date_time}')
    print(f'saving to: {result_dir_timed}')
    os.makedirs(result_dir_timed, exist_ok=True)

    data = load_data(basedir, date, dataset_number)

    hw_2(result_dir_timed, data, sigma)
