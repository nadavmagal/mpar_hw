from FAST.read_data import read_world, read_sensor_data
from FAST.misc_tools import *
import numpy as np
import math
import copy

# plot preferences, interactive plotting mode
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()


def initialize_particles(num_particles, num_landmarks):
    # initialize particle at pose [0,0,0] with an empty map

    particles = []

    for i in range(num_particles):
        particle = dict()

        # initialize pose: at the beginning, robot is certain it is at [0,0,0]
        particle['x'] = 0
        particle['y'] = 0
        particle['theta'] = 0

        # initial weight
        particle['weight'] = 1.0 / num_particles

        # particle history aka all visited poses
        particle['history'] = []

        # initialize landmarks of the particle
        landmarks = dict()

        for i in range(num_landmarks):
            landmark = dict()

            # initialize the landmark mean and covariance
            landmark['mu'] = [0, 0]
            landmark['sigma'] = np.zeros([2, 2])
            landmark['observed'] = False

            landmarks[i + 1] = landmark

        # add landmarks to particle
        particle['landmarks'] = landmarks

        # add particle to set
        particles.append(particle)

    return particles


def get_normalized_angle(theta):
    while theta > np.pi:
        theta = theta - 2 * np.pi

    while theta < -np.pi:
        theta = theta + 2 * np.pi

    return theta


def sample_motion_model(odometry, particles):
    # Updates the particle positions, based on old positions, the odometry
    # measurements and the motion noise 

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    sigma_rot1, sigma_trans, sigma_rot2 = [0.01, 0.02, 0.01]

    '''your code here'''
    for cur_particle in particles:
        cur_particle['history'].append([cur_particle['x'],cur_particle['y'],cur_particle['theta']])

        cur_rot1 = np.random.normal(delta_rot1, sigma_rot1)
        cur_trans = np.random.normal(delta_trans, sigma_trans)
        cur_rot2 = np.random.normal(delta_rot2, sigma_rot2)

        cur_particle['x'] += cur_trans * np.cos(cur_particle['theta'] + cur_rot1)
        cur_particle['y'] += cur_trans * np.sin(cur_particle['theta'] + cur_rot1)
        cur_particle['theta'] += cur_rot1 + cur_rot2
        cur_particle['theta'] = get_normalized_angle(cur_particle['theta'])

    '''***        ***'''
    return


def measurement_model(particle, landmark):
    # Compute the expected measurement for a landmark
    # and the Jacobian with respect to the landmark.

    px = particle['x']
    py = particle['y']
    ptheta = particle['theta']

    lx = landmark['mu'][0]
    ly = landmark['mu'][1]

    # calculate expected range measurement
    meas_range_exp = np.sqrt((lx - px) ** 2 + (ly - py) ** 2)
    meas_bearing_exp = math.atan2(ly - py, lx - px) - ptheta

    h = np.array([meas_range_exp, meas_bearing_exp])

    # Compute the Jacobian H of the measurement function h 
    # wrt the landmark location

    H = np.zeros((2, 2))
    H[0, 0] = (lx - px) / h[0]
    H[0, 1] = (ly - py) / h[0]
    H[1, 0] = (py - ly) / (h[0] ** 2)
    H[1, 1] = (lx - px) / (h[0] ** 2)

    return h, H


def eval_sensor_model(sensor_data, particles):
    # Correct landmark poses with a measurement and
    # calculate particle weight

    # sensor noise
    Q_t = np.array([[0.1, 0], \
                    [0, 0.1]])

    # measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    bearings = sensor_data['bearing']

    # update landmarks and calculate weight for each particle
    for particle in particles:

        landmarks = particle['landmarks']

        px = particle['x']
        py = particle['y']
        ptheta = particle['theta']

        # loop over observed landmarks
        for i in range(len(ids)):

            # current landmark
            lm_id = ids[i]
            landmark = landmarks[lm_id]

            # measured range and bearing to current landmark
            meas_range = ranges[i]
            meas_bearing = bearings[i]

            if not landmark['observed']:
                # landmark is observed for the first time

                # initialize landmark mean and covariance. You can use the
                # provided function 'measurement_model' above
                '''your code here'''
                h, H = measurement_model(particle, landmark)
                landmark['sigma'] = np.linalg.inv(H) @ Q_t @ np.linalg.inv(H).T  # TODO: pay attention
                '''***        ***'''

                landmark['observed'] = True

            else:
                # landmark was observed before

                # update landmark mean and covariance. You can use the
                # provided function 'measurement_model' above. 
                # calculate particle weight: particle['weight'] = ...
                '''your code here'''
                h, H = measurement_model(particle, landmark)
                Q = H @ landmark['sigma'] * H.T + Q_t
                K = landmark['sigma'] @ H.T @ np.linalg.inv(Q)

                diff = np.array([meas_range, meas_bearing]) - h
                diff[1] = get_normalized_angle(diff[1])
                landmark['mu'] += K @ diff
                landmark['sigma'] += (np.eye(2) - K @ H) @ landmark['sigma']
                particle['weight'] = particle['weight'] * (1/np.sqrt(np.linalg.norm(2*np.pi*Q)))*np.exp(-0.5*diff.T@np.linalg.inv(Q)@diff)
                '''***        ***'''

    # normalize weights
    normalizer = sum([p['weight'] for p in particles])

    for particle in particles:
        particle['weight'] = particle['weight'] / normalizer

    return


def resample_particles(particles):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle 
    # weights.

    new_particles = []

    '''your code here'''
    for cur_particle in particles:
        cur_new_particle = copy.deepcopy(cur_particle)

        new_particles.append(cur_new_particle)

    '''***        ***'''

    # hint: To copy a particle from particles to the new_particles
    # list, first make a copy:
    # new_particle = copy.deepcopy(particles[i])
    # ...
    # new_particles.append(new_particle)

    return new_particles


def get_odometry_trajectory(sensor_readings):
    num_of_samples = int(len(sensor_readings.keys()) / 2)
    x = [0]
    y = [0]
    theta = [0]
    for ii in range(num_of_samples):
        delta_rot_1 = sensor_readings[(ii, 'odometry')]['r1']
        delta_rot_2 = sensor_readings[(ii, 'odometry')]['r2']
        delta_trans = sensor_readings[(ii, 'odometry')]['t']

        x.append(x[-1] + delta_trans * np.cos(theta[-1] + delta_rot_1))
        y.append(y[-1] + delta_trans * np.sin(theta[-1] + delta_rot_1))
        theta.append(theta[-1] + delta_rot_1 + delta_rot_2)
    trajectory = np.vstack([x, y, theta]).T

    return trajectory


def main():
    print("Reading landmark positions")
    landmarks = read_world(
        "/home/nadav/studies/mapping_and_perception_autonomous_robots/mpar_hw_code/project3_code/FAST/world.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data(
        "/home/nadav/studies/mapping_and_perception_autonomous_robots/mpar_hw_code/project3_code/FAST/sensor_data.dat")

    num_particles = 100
    num_landmarks = len(landmarks)

    # create particle set
    particles = initialize_particles(num_particles, num_landmarks)

    gt_trajectory = get_odometry_trajectory(sensor_readings)

    # run FastSLAM
    for timestep in range(int(len(sensor_readings) / 2)):
        if False:
            plot_fastSlam(gt_trajectory, landmarks, particles, timestep)
        # predict particles by sampling from motion model with odometry info
        sample_motion_model(sensor_readings[timestep, 'odometry'], particles)

        # evaluate sensor model to update landmarks and calculate particle weights
        eval_sensor_model(sensor_readings[timestep, 'sensor'], particles)

        # plot filter state
        plot_state(particles, landmarks)

        # calculate new set of equally weighted particles
        particles = resample_particles(particles)

    plt.show('hold')


def plot_fastSlam(gt_trajectory, landmarks, particles, timestep):
    plt.figure()
    for cur_lm in landmarks:
        plt.scatter(landmarks[cur_lm][0], landmarks[cur_lm][1], color='black')
    plt.scatter([], [], color='black', label='landmarks')
    plt.scatter(gt_trajectory[:timestep + 1, 0], gt_trajectory[:timestep + 1, 1], color='blue', label='GT trajectory')
    for cur_particle in particles:
        plt.scatter(cur_particle['x'], cur_particle['y'], color='magenta', s=1)
    plt.scatter([], [], color='magenta', s=1, label='particles')
    plt.legend()
    plt.grid()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.show(block=False)


if __name__ == "__main__":
    main()
