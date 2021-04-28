import numpy as np


def read_world(file_name):
    file1 = open(file_name, 'r')
    Lines = file1.readlines()

    landmarks = []
    for line in Lines:
        cur_lendmark = np.array([float(cur_digit) for cur_digit in line.split(' ')])
        landmarks.append(cur_lendmark)
    return np.array(landmarks)


def read_data(file_name, noise=None):
    '''
    % Reads the odometry and sensor readings from a file.
    %
    % filename: path to the file to parse
    % data: structure containing the parsed information
    %
    % The data is returned in a structure where the u_t and z_t are stored
    % within a single entry. A z_t can contain observations of multiple
    % landmarks.
    %
    % Usage:
    % - access the readings for timestep i:
    %   data.timestep(i)
    %   this returns a structure containing the odometry reading and all
    %   landmark obsevations, which can be accessed as follows
    % - odometry reading at timestep i:
    %   data.timestep(i).odometry
    % - senor reading at timestep i:
    %   data.timestep(i).sensor
    %
    % Odometry readings have the following fields:
    % - r1 : rotation 1
    % - t  : translation
    % - r2 : rotation 2
    % which correspond to the identically labeled variables in the motion
    % mode.
    %
    % Sensor readings can again be indexed and each of the entris has the
    % following fields:
    % - id      : id of the observed landmark
    % - range   : measured range to the landmark
    % - bearing : measured angle to the landmark (you can ignore this)
    %
    % Examples:
    % - Translational component of the odometry reading at timestep 10
    %   data.timestep(10).odometry.t
    % - Measured range to the second landmark observed at timestep 4
    %   data.timestep(4).sensor(2).range
    '''
    file1 = open(file_name, 'r')
    Lines = file1.readlines()

    data = {'timestep': []}
    T = 0
    for line in Lines:
        arr = line.split(' ')
        type = arr[0]

        if type == 'ODOMETRY':
            if T > 0:
                data['timestep'].append(cur_timestep)
            T += 1
            cur_timestep = {
                'odometry': dict(),
                'sensor': []}
            cur_timestep['odometry']['r1'] = float(arr[1]) + np.random.normal(0, noise['sigme_rot1'], 1)[0]
            cur_timestep['odometry']['t'] = float(arr[2])+ np.random.normal(0, noise['sigme_t'], 1)[0]
            cur_timestep['odometry']['r2'] = float(arr[3])+ np.random.normal(0, noise['sigme_rot2'], 1)[0]
            a = 3
        elif type == 'SENSOR':
            cur_sensor = {
                'id': int(arr[1]),
                'range': float(arr[2]),
                'bearing': float(arr[3])
            }
            cur_timestep['sensor'].append(cur_sensor)
            a = 3
    data['timestep'].append(cur_timestep)
    # data['timestep'] = data['timestep'][1::]
    a = 3
    return data


def normalize_angle(phi):
    if phi > np.pi:
        phi = phi - 2 * np.pi
    if phi < -np.pi:
        phi = phi + 2 * np.pi
    return phi


def normalize_all_bearings(z):
    # % Go over the observations vector and normalize the bearings
    # % The expected format of z is [range; bearing; range; bearing; ...]
    # for i=2:2:length(z):
    for ii in list(range(1, len(z), 2)):
        z[ii] = normalize_angle(z[ii])
    zNorm = z
    return zNorm
