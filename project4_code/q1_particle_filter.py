import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from read_data import read_world, read_sensor_data
import time

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
        return

    def plot_state(self, ii, landmarks, save_plot=False, show_plot=False, save_path=None):
        if False:
            plt.figure()
            plt.scatter(range(self.weights.shape[0]), self.weights)
            plt.show(block=False)

        plt.figure()
        plt.scatter(landmarks[:, 0], landmarks[:, 1], color='blue', label='landmarks')
        plt.scatter(self.particles[:, 0], self.particles[:, 1], color='red', label='particles')
        for ll in range(self.weights.shape[0]):
            plt.text(self.particles[ll, 0], self.particles[ll, 1], round(self.weights[ll], 2))
        plt.legend()
        plt.grid()
        plt.title('particles distribution')
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')

        if show_plot:
            plt.show(block=False)
        if save_plot:
            plt.savefig(os.path.join(save_path, f'state_{ii}.png'), dpi=150)

    def prediction(self, odometry):
        self.particles_history.append(self.particles.copy())
        delta_rot_1 = np.random.normal(odometry[0], self.sigmot['sigma_rot1'])
        delta_rot_2 = np.random.normal(odometry[2], self.sigmot['sigma_rot2'])
        delta_trans = np.random.normal(odometry[1], self.sigmot['sigma_trans'])

        for jj, cur_particle in enumerate(self.particles):
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


def run_particle_filter(landmarks, odometry, lidar_measurments, N, save_path):
    particle_filter = ParticleFilter(N)

    if False:
        plt.figure()
        plt.scatter(landmarks[:, 0], landmarks[:, 1], color='blue', label='landmarks')
        plt.scatter(particle_filter.get_particles()[:, 0], particle_filter.get_particles()[:, 1], color='red',
                    label='particles')
        plt.legend()
        plt.grid()
        plt.title('initial particles distribution')
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        plt.show(block=False)

    for ii in tqdm.tqdm(range(odometry.shape[0])):
        cur_odometry = odometry[ii]
        cur_lidar_measurment = lidar_measurments[ii]
        particle_filter.prediction(cur_odometry)
        particle_filter.eval_sensor_model(cur_lidar_measurment, landmarks)
        particle_filter.plot_state(ii, landmarks, save_plot=True, show_plot=True, save_path=save_path)

    a = 3


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

    landmarks = np.array(pd.read_csv(landmarks_fp, header=None))
    odometry = np.array(pd.read_csv(odometry_fp, header=None, delimiter=' '))[:, 1:]  # r1, trans, r2

    gt_trajectory = get_odometry_trajectory(odometry)

    if True:
        plt.figure()
        plt.scatter(gt_trajectory[:, 0], gt_trajectory[:, 1], color='black', label='GT trajectory')
        plt.scatter(landmarks[:, 0], landmarks[:, 1], color='blue', label='landmarks')
        plt.legend()
        plt.grid()
        plt.title('GT trajecory and landmarks')
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        plt.show(block=False)

    lidar_measurments = create_lidar_measurments(gt_trajectory, landmarks)

    N = 100  # number of particles
    run_particle_filter(landmarks, odometry, lidar_measurments, N, save_path)


if __name__ == "__main__":
    np.random.seed(0)
    main()
