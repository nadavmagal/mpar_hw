import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from read_data import read_world, read_sensor_data
import time
import os


class ParticleFilter:
    def __init__(self, N):
        self.particles = self._init_particles(N)
        self.weights = np.ones(N) / N
        self.particles_history = []

        self.sigmot = {
            'sigma_rot1': 0.01,
            'sigma_trans': 0.04,
            'sigma_rot2': 0.01
        }

    @staticmethod
    def _init_particles(N):
        length = 10 * np.sqrt(np.random.uniform(0, 1, N))
        angle = np.pi * np.random.uniform(0, 2, N)
        particles = np.vstack([length * np.cos(angle) + 5, length * np.sin(angle) + 5, np.zeros(N)]).T
        return particles

    def get_particles(self):
        return self.particles

    def eval_sensor_model(self, lidar_measurements, landmarks):
        for jj, cur_particle in enumerate(self.particles):
            self.weights[jj] = self.calculate_patricle_weight(cur_particle, landmarks, lidar_measurements)

        normalizer = sum(self.weights)
        for jj in range(self.weights.shape[0]):
            self.weights[jj] = self.weights[jj] / normalizer

        return

    def resample_particles(self):
        num_of_particles = self.particles.shape[0]
        new_particles = []
        r = (1 / num_of_particles) * np.random.uniform()
        c = self.weights[0]
        ii = 0
        for particle_counter in range(num_of_particles):
            U = r + (particle_counter) / num_of_particles
            while U > c:
                ii = ii + 1
                c = c + self.weights[ii]

            new_particles.append(self.particles[ii].copy())

        self.particles = np.array(new_particles)
        return

    def plot_state(self, ii, landmarks, gt_trajectory, best_results_arr, average_results, save_plot=False,
                   show_plot=False, save_path=None):
        if False:
            plt.figure()
            plt.scatter(range(self.weights.shape[0]), self.weights)
            plt.show(block=False)

        if show_plot or save_plot:
            plt.figure()
            plt.scatter(gt_trajectory[:, 0], gt_trajectory[:, 1], s=2, color='black', label='GT trajectory')
            plt.scatter(best_results_arr[:, 0], best_results_arr[:, 1], s=2, color='green', label='best trajectory')
            plt.scatter(average_results[:, 0], average_results[:, 1], s=2, color='blue', label='ave trajectory')
            plt.scatter(landmarks[:, 0], landmarks[:, 1], s=2, color='magenta', label='landmarks')
            plt.scatter(self.particles[:, 0], self.particles[:, 1], s=2, color='red', label='particles')
            # for ll in range(self.weights.shape[0]):
            #     plt.text(self.particles[ll, 0], self.particles[ll, 1], round(self.weights[ll], 2))
            plt.legend()
            plt.grid()
            plt.title('particles distribution')
            plt.xlabel('x[m]')
            plt.ylabel('y[m]')

        if show_plot:
            plt.show(block=False)
        if save_plot:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f'state_{ii}.png'), dpi=150)

    def prediction(self, odometry):
        self.particles_history.append(self.particles.copy())

        for jj, cur_particle in enumerate(self.particles):
            delta_rot_1 = np.random.normal(odometry[0], self.sigmot['sigma_rot1'])
            delta_rot_2 = np.random.normal(odometry[2], self.sigmot['sigma_rot2'])
            delta_trans = np.random.normal(odometry[1], self.sigmot['sigma_trans'])
            cur_particle[0] = cur_particle[0] + delta_trans * np.cos(cur_particle[2] + delta_rot_1)
            cur_particle[1] = cur_particle[1] + delta_trans * np.sin(cur_particle[2] + delta_rot_1)
            cur_particle[2] = self.normalize_angle(cur_particle[2] + delta_rot_1 + delta_rot_2)

        return

    @staticmethod
    def calculate_patricle_weight(particle, landmarks, lidar_measurments):
        N_landmarks = landmarks.shape[0]
        patricle_distance_from_landmarks = np.linalg.norm(np.tile(particle[0:2], [N_landmarks, 1]) - landmarks, axis=1)
        rmse = np.linalg.norm(patricle_distance_from_landmarks - lidar_measurments)

        coef = 0.1
        weight = np.exp(-coef * rmse)

        return weight

    @staticmethod
    def normalize_angle(angle):
        while angle > np.pi:
            angle = angle - 2 * np.pi

        while angle < -np.pi:
            angle = angle + 2 * np.pi

        return angle

    def get_best_particle_pose(self):
        index_of_best_particle = np.argmax(self.weights)
        return self.particles[index_of_best_particle, 0:2]

    def get_w_average_pose(self):
        ave_x = np.sum(np.multiply(self.weights, self.particles[:, 0])) / np.sum(self.weights)
        ave_y = np.sum(np.multiply(self.weights, self.particles[:, 1])) / np.sum(self.weights)
        return np.array([ave_x, ave_y])


def get_odometry_trajectory(odometry):
    num_of_samples = odometry.shape[0]
    x = [0]
    y = [0]
    theta = [0]
    for ii in range(num_of_samples):
        # delta_rot_1 = sensor_data[(ii, 'odometry')]['r1']
        # delta_rot_2 = sensor_data[(ii, 'odometry')]['r2']
        # delta_trans = sensor_data[(ii, 'odometry')]['t']
        delta_rot_1 = odometry[ii, 0]
        delta_rot_2 = odometry[ii, 2]
        delta_trans = odometry[ii, 1]

        x.append(x[-1] + delta_trans * np.cos(theta[-1] + delta_rot_1))
        y.append(y[-1] + delta_trans * np.sin(theta[-1] + delta_rot_1))
        theta.append(theta[-1] + delta_rot_1 + delta_rot_2)
    trajectory = np.vstack([x, y, theta]).T

    return trajectory

def get_rmse(gt_trajectory, average_results_arr, best_results_arr):
    diff_best = gt_trajectory[:best_results_arr.shape[0],0:2] - best_results_arr
    diff_ave = gt_trajectory[:average_results_arr.shape[0],0:2] - average_results_arr

    best_rmse = np.sqrt(np.sum(np.power(diff_best, 2)) / diff_best.shape[0])
    ave_rmse = np.sqrt(np.sum(np.power(diff_ave, 2)) / diff_ave.shape[0])

    return best_rmse, ave_rmse

def run_particle_filter(landmarks, odometry, lidar_measurments, N, timed_save_path, gt_trajectory, save_plot=False):
    particle_filter = ParticleFilter(N)

    best_results = []
    average_results = []
    for ii in tqdm.tqdm(range(odometry.shape[0])):
        cur_odometry = odometry[ii]
        cur_lidar_measurment = lidar_measurments[ii]

        particle_filter.prediction(cur_odometry)
        particle_filter.eval_sensor_model(cur_lidar_measurment, landmarks)

        best_results.append(particle_filter.get_best_particle_pose())
        average_results.append(particle_filter.get_w_average_pose())
        best_results_arr = np.array(best_results)
        average_results_arr = np.array(average_results)

        particle_filter.plot_state(ii, landmarks, gt_trajectory, best_results_arr, average_results_arr, save_plot=save_plot,
                                   show_plot=False,
                                   save_path=timed_save_path)
        particle_filter.resample_particles()

    best_trajectory_rmse, average_trajectory_rmse = get_rmse(gt_trajectory, average_results_arr, best_results_arr)

    return best_trajectory_rmse, average_trajectory_rmse


def create_lidar_measurments(gt_trajectory, landmarks):
    lidar_measurments = []
    N_landmarks = landmarks.shape[0]
    for cur_position in gt_trajectory:
        cur_measurment = np.linalg.norm(np.tile(cur_position[0:2], [N_landmarks, 1]) - landmarks, axis=1)
        lidar_measurments.append(cur_measurment)
    lidar_measurments = np.array(lidar_measurments)
    return lidar_measurments


def main():
    landmarks_fp = r'/home/nadav/studies/mapping_and_perception_autonomous_robots/project_4/files/landmarks_EX1.csv'
    odometry_fp = r'/home/nadav/studies/mapping_and_perception_autonomous_robots/project_4/files/odometry.dat'

    save_path = r'/home/nadav/studies/mapping_and_perception_autonomous_robots/project_4/results'
    date_and_time = time.strftime("%Y.%m.%d-%H.%M")
    timed_save_path = os.path.join(save_path, date_and_time)


    landmarks = np.array(pd.read_csv(landmarks_fp, header=None))
    odometry = np.array(pd.read_csv(odometry_fp, header=None, delimiter=' '))[:, 1:]  # r1, trans, r2

    gt_trajectory = get_odometry_trajectory(odometry)
    lidar_measurments = create_lidar_measurments(gt_trajectory, landmarks)

    N = 100  # number of particles
    total_best_trajectory_rmse = []
    total_average_trajectory_rmse = []
    N_vec = list(range(100,0,-1))
    # N_vec = [100]
    for N in N_vec:
        timed_save_path_with_N = timed_save_path+f'__N-{N}'
        best_trajectory_rmse, average_trajectory_rmse = run_particle_filter(landmarks, odometry, lidar_measurments, N, timed_save_path_with_N, gt_trajectory, save_plot=False)
        total_best_trajectory_rmse.append(best_trajectory_rmse)
        total_average_trajectory_rmse.append(average_trajectory_rmse)

    plt.figure()
    plt.scatter(N_vec, total_best_trajectory_rmse, label='rmse best')
    plt.scatter(N_vec, total_average_trajectory_rmse, label='rmse average')
    plt.grid()
    plt.title("RMSE of best and average particle trajectory as function of number of particles")
    plt.xlabel('N')
    plt.ylabel('rmse [m]')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'rmse.png'), dpi=150)

if __name__ == "__main__":
    np.random.seed(0)
    main()

    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure()
    x = np.array(list(range(0,50)))
    plt.plot(x, np.exp(-0.05*x))
    plt.grid()
    plt.title('exp(-0.05x)')
    plt.show(block=False)
