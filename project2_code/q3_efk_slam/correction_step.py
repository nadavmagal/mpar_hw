import numpy as np
from tools import normalize_angle, normalize_all_bearings


def correction_step(mu, sigma, z, observedLandmarks, sigmot):
    a = 3
    '''
    % Updates the belief, i. e., mu and sigma after observing landmarks, according to the sensor model
    % The employed sensor model measures the range and bearing of a landmark
    % mu: 2N+3 x 1 vector representing the state mean.
    % The first 3 components of mu correspond to the current estimate of the robot pose [x; y; theta]
    % The current pose estimate of the landmark with id = j is: [mu(2*j+2); mu(2*j+3)]
    % sigma: 2N+3 x 2N+3 is the covariance matrix
    % z: struct array containing the landmark observations.
    % Each observation z(i) has an id z(i).id, a range z(i).range, and a bearing z(i).bearing
    % The vector observedLandmarks indicates which landmarks have been observed
    % at some point by the robot.
    % observedLandmarks(j) is false if the landmark with id = j has never been observed before.
    '''
    # % Number of measurements in this time step
    m = len(z)
    # % Number of dimensions to mu
    dim = mu.shape[0]
    '''
    % Z: vectorized form of all measurements made in this time step: [range_1; bearing_1; range_2; bearing_2; ...; range_m; bearing_m]
    % ExpectedZ: vectorized form of all expected measurements in the same form.
    % They are initialized here and should be filled out in the for loop below
    '''
    Z = np.zeros(m * 2)
    expectedZ = np.zeros(m * 2)

    '''
    % Iterate over the measurements and compute the H matrix
    % (stacked Jacobian blocks of the measurement function)
    % H will be 2m x 2N+3
    '''
    for ii in range(m):
        # % Get the id of the landmark corresponding to the i-th observation
        landmarkId = z[ii]['id']

        # % If the landmark is obeserved for the first time:

        if observedLandmarks[landmarkId - 1] == False:
            # % TODO: Initialize its pose in mu based on the measurement and the current robot pose:
            mu[2 * landmarkId + 1] = mu[0] + z[ii]['range'] * np.cos(z[ii]['bearing'] + mu[2])
            mu[2 * landmarkId + 2] = mu[1] + z[ii]['range'] * np.sin(z[ii]['bearing'] + mu[2])
            observedLandmarks[landmarkId - 1] = True

        # % TODO: Add the landmark measurement to the Z vector
        Z[2 * ii: 2 * ii + 2] = [z[ii]['range'], z[ii]['bearing']]

        # % TODO: Use the current estimate of the landmark pose
        # % to compute the corresponding expected measurement in expectedZ:
        delta = mu[2 * landmarkId + 1: 2 * landmarkId + 3] - mu[0: 2]
        q = delta.T @ delta
        expectedZ[2 * ii:2 * ii + 2] = [np.sqrt(q), normalize_angle(np.arctan2(delta[1], delta[0]) - mu[2])]

        # % TODO: Compute the Jacobian Hi of the measurement function h for this observation
        # % Map Jacobian Hi to high dimensional space by a mapping matrix Fxj
        Fxj = np.zeros([5, dim])
        Fxj[0:3, 0:3] = np.eye(3)
        Fxj[3, 2 * landmarkId + 1] = 1.
        Fxj[4, 2 * landmarkId + 2] = 1.

        Hi = (1 / q) * np.array(
            [[-np.sqrt(q) * delta[0], -np.sqrt(q) * delta[1], 0, np.sqrt(q) * delta[0], np.sqrt(q) * delta[1]],
             [delta[1], -delta[0], -q, -delta[1], delta[0]]])

        Hi = Hi @ Fxj
        # % Augment H with the new Hi
        if ii == 0:
            H = Hi
        else:
            H = np.vstack([H, Hi])

    # % TODO: Construct the sensor noise matrix Q
    sigma_r_squar, sigma_phi_squar = sigmot['sigma_r_squar'], sigmot['sigma_phi_squar']
    Q = np.diag([sigma_r_squar, sigma_phi_squar]*m)
    # % TODO: Compute the Kalman gain
    K = sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + Q)

    # % TODO: Compute the difference between the expected and recorded measurements.
    # % Remember to normalize the bearings after subtracting!
    # % (hint: use the normalize_all_bearings function available in tools)
    delta_Z = normalize_all_bearings(Z - expectedZ)

    # % TODO: Finish the correction step by computing the new mu and sigma.
    # % Normalize theta in the robot pose.
    mu = mu + K @ delta_Z
    sigma = (np.eye(dim) - K @ H) @ sigma
    return mu, sigma, observedLandmarks
