from FAST.read_data import read_world, read_sensor_data
from FAST.maya.misc_tools import *
import numpy as np
import math
import copy
from numpy.linalg import inv
from numpy.linalg import multi_dot
import random

#plot preferences, interactive plotting mode
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()


def initialize_particles(num_particles, num_landmarks):
    #initialize particle at pose [0,0,0] with an empty map

    particles = []

    for i in range(num_particles):
        particle = dict()

        #initialize pose: at the beginning, robot is certain it is at [0,0,0]
        particle['x'] = 0
        particle['y'] = 0
        particle['theta'] = 0

        #initial weight
        particle['weight'] = 1.0 / num_particles
        
        #particle history aka all visited poses
        particle['history'] = []

        #initialize landmarks of the particle
        landmarks = dict()

        for i in range(num_landmarks):
            landmark = dict()

            #initialize the landmark mean and covariance 
            landmark['mu'] = [0,0]
            landmark['sigma'] = np.zeros([2,2])
            landmark['observed'] = False

            landmarks[i+1] = landmark

        #add landmarks to particle
        particle['landmarks'] = landmarks

        #add particle to set
        particles.append(particle)

    return particles


def sample_motion_model(odometry, particles):
    # Updates the particle positions, based on old positions, the odometry
    # measurements and the motion noise 

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]

    '''your code here'''
    '''***        ***'''
    num_particles = len(particles)
    # sig_r1 = 0.01
    # sig_t = 0.02
    # sig_r2 = 0.01

    sig_r1 = 0.0
    sig_t = 0.0
    sig_r2 = 0.0

    sigma_r1 = np.random.normal(0, sig_r1, num_particles)
    sigma_t = np.random.normal(0, sig_t, num_particles)
    sigma_r2 = np.random.normal(0, sig_r2, num_particles)

    delta_rot1 = delta_rot1 + sigma_r1
    delta_trans = delta_trans + sigma_t
    delta_rot2 = delta_rot2 + sigma_r2

    for i in range(num_particles):
        particles[i]['x'] = particles[i]['x'] + delta_trans[i] * np.cos(particles[i]['theta'] + delta_rot1[i])
        particles[i]['y'] = particles[i]['y'] + delta_trans[i] * np.sin(particles[i]['theta'] + delta_rot1[i])
        particles[i]['theta'] = particles[i]['theta'] + delta_rot1[i]+delta_rot2[i]


def measurement_model(particle, landmark):
    #Compute the expected measurement for a landmark
    #and the Jacobian with respect to the landmark.

    px = particle['x']
    py = particle['y']
    ptheta = particle['theta']

    lx = landmark['mu'][0]
    ly = landmark['mu'][1]

    #calculate expected range measurement
    meas_range_exp = np.sqrt((lx - px)**2 + (ly - py)**2)
    meas_bearing_exp = math.atan2(ly - py, lx - px) - ptheta

    h = np.array([meas_range_exp, meas_bearing_exp])

    # Compute the Jacobian H of the measurement function h 
    #wrt the landmark location
    
    H = np.zeros((2,2))
    H[0,0] = (lx - px) / h[0]
    H[0,1] = (ly - py) / h[0]
    H[1,0] = (py - ly) / (h[0]**2)
    H[1,1] = (lx - px) / (h[0]**2)

    return h, H


def eval_sensor_model(sensor_data, particles):
    #Correct landmark poses with a measurement and
    #calculate particle weight

    #sensor noise
    #Q_t = np.array([[0.1, 0],\
    #               [0, 0.1]])

    Q_t = np.array([[1, 0],\
                    [0, 0.1]])
    #measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    bearings = sensor_data['bearing']

    #update landmarks and calculate weight for each particle
    for particle in particles:

        landmarks = particle['landmarks']

        px = particle['x']
        py = particle['y']
        ptheta = particle['theta'] 

        #loop over observed landmarks 
        for i in range(len(ids)):

            #current landmark
            lm_id = ids[i]
            landmark = landmarks[lm_id]
            
            #measured range and bearing to current landmark
            meas_range = ranges[i]
            meas_bearing = bearings[i]

            if not landmark['observed']:
                # landmark is observed for the first time
                
                # initialize landmark mean and covariance. You can use the
                # provided function 'measurement_model' above
                '''your code here'''
                '''***        ***'''
                z = np.array([meas_range, meas_bearing])
                landmark['mu'][0] = px + z[0] * np.cos(z[1] + ptheta)
                landmark['mu'][1] = py + z[0] * np.sin(z[1] + ptheta)
                h, H = measurement_model(particle, landmark)
                #h[1]=normalize_angle(h[1])
                landmark['sigma'] = multi_dot([inv(H), Q_t, inv(H).T])
                landmark['observed'] = True

            else:
                # landmark was observed before

                # update landmark mean and covariance. You can use the
                # provided function 'measurement_model' above. 
                # calculate particle weight: particle['weight'] = ...
                '''your code here'''
                '''***        ***'''
                h, H = measurement_model(particle, landmark)
                h[1] = normalize_angle(h[1])
                z = np.array([meas_range, meas_bearing])
                z_exp = h
                Q = H@ landmark['sigma']@ H.T+Q_t
                K = landmark['sigma']@H.T@inv(Q)
                landmark['mu'] = landmark['mu']+np.dot(K, (z-z_exp).T)
                landmark['sigma'] = (np.eye(2)-np.dot(K, H))@landmark['sigma']
                diff = np.array([z[0]-z_exp[0], angle_diff(z[1], z_exp[1])])
                particle['weight'] = (np.linalg.det(2*np.pi*Q))**(-0.5)*np.exp(-0.5*(diff@inv(Q)@diff.T))


    #normalize weights
    normalizer = sum([p['weight'] for p in particles])

    for particle in particles:
        particle['weight'] = particle['weight'] / normalizer


def resample_particles(particles):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle 
    # weights.

    new_particles = []

    '''your code here'''
    '''***        ***'''

    # hint: To copy a particle from particles to the new_particles
    # list, first make a copy:
    # new_particle = copy.deepcopy(particles[i])
    # ...
    # new_particles.append(new_particle)
    J = len(particles)
    r = random.random()/J
    c = particles[0]['weight']
    i = 0
    for j in range(J):
        U = r+j/J
        while U > c:
            i = i+1
            c = c+particles[i]['weight']
        new_particle = copy.deepcopy(particles[i])
        new_particles.append(new_particle)
    return new_particles


def sample_motion_model_gt(odometry, particles):
    # Updates the particle positions, based on old positions, the odometry
    # measurements and the motion noise

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    '''your code here'''
    '''***        ***'''
    num_particles = len(particles)

    particles[0]['x'] = particles[0]['x'] + delta_trans * np.cos(particles[0]['theta'] + delta_rot1)
    particles[0]['y'] = particles[0]['y'] + delta_trans * np.sin(particles[0]['theta'] + delta_rot1)
    particles[0]['theta'] = particles[0]['theta'] + delta_rot1 + delta_rot2

def main():

    print("Reading landmark positions")
    landmarks = read_world(
        "/home/nadav/studies/mapping_and_perception_autonomous_robots/mpar_hw_code/project3_code/FAST/world.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data(
        "/home/nadav/studies/mapping_and_perception_autonomous_robots/mpar_hw_code/project3_code/FAST/sensor_data.dat")
    num_particles = 100
    num_landmarks = len(landmarks)

    #create particle set
    particles = initialize_particles(num_particles, num_landmarks)
    particles_gt = initialize_particles(1, num_landmarks)

    path_gt = np.zeros((2,int(len(sensor_readings)/2)))
    path_exp_avg = np.zeros((2, int(len(sensor_readings) / 2)))
    path_exp_best = np.zeros((2, int(len(sensor_readings) / 2)))

    #run FastSLAM
    for timestep in range(int(len(sensor_readings)/2)):

        #predict particles by sampling from motion model with odometry info
        sample_motion_model(sensor_readings[timestep,'odometry'], particles)
        sample_motion_model_gt(sensor_readings[timestep, 'odometry'], particles_gt)

        path_gt[:, timestep] = np.array([particles_gt[0]['x'], particles_gt[0]['y']])
        path_exp_best[:, timestep] = np.array([best_particle(particles)['x'], best_particle(particles)['y']])
        path_exp_avg[:, timestep] = np.array([avg_particle(particles)[0], avg_particle(particles)[1]])

        #evaluate sensor model to update landmarks and calculate particle weights
        eval_sensor_model(sensor_readings[timestep, 'sensor'], particles)

        #plot filter state
        plot_state(particles, landmarks)
        plt.plot(path_gt[0, :], path_gt[1, :])
        plt.plot(path_exp_best[0, :], path_exp_best[1, :], '.')
        plt.plot(path_exp_avg[0, :], path_exp_avg[1, :], '.')
        #calculate new set of equally weighted particles
        particles = resample_particles(particles)

    plt.show('hold')

if __name__ == "__main__":
    np.random.seed(0)
    main()