import numpy as np


def prediction_step(mu, sigma, u):
    a = 3
    '''
    % Updates the belief concerning the robot pose according to the motion model,
    % mu: 2N+3 x 1 vector representing the state mean
    % sigma: 2N+3 x 2N+3 covariance matrix
    % u: odometry reading (r1, t, r2)
    % Use u.r1, u.t, and u.r2 to access the rotation and translation values

    [m,n] = size(sigma);

    v = zeros(m,1);
    % TODO: Compute the new mu based on the noise-free (odometry-based) motion model
    % Remember to normalize theta after the update (hint: use the function normalize_angle available in tools)

    % % TODO: Construct the full Jacobian G

    G = [Gx, zeros(3, n-3); zeros(n-3, 3), eye(n-3)];

    % % TODO: Motion noise R

    % Compute the predicted sigma after incorporating the motion
    sigma = G*sigma*G' + R;
    '''
    m, n = sigma.shape
    prev_theta = mu[2]
    Fx = np.hstack([np.eye(3), np.zeros([3, mu.shape[0] - 3])])
    mu += Fx.T @ np.array([u['t'] * np.cos(prev_theta + u['r1']),
                           u['t'] * np.sin(prev_theta + u['r1']),
                           u['r1'] + u['r2']])
    mu[2] = normalize_angle(mu[2])

    # % TODO: Compute the 3x3 Jacobian Gx of the motion model

    Gx = np.eye(3)
    Gx[0, 2] = -u['t'] * np.sin(prev_theta + u['r1'])
    Gx[1, 2] = u['t'] * np.cos(prev_theta + u['r1'])

    G = np.vstack([np.hstack([Gx, np.zeros([3, n - 3])]), np.hstack([np.zeros([n - 3, 3]), np.eye(n - 3)])])

    # % TODO: Compute the 3x3 Jacobian Rx of the motion model
    sigma_rot1, sigma_t, sigma_rot2 = 0.1, 0.1, 0.1
    R_tilda = np.diag([sigma_rot1 ** 2, sigma_t ** 2, sigma_rot2 ** 2])
    V = np.zeros([3, 3])
    V[0, 0] = -u['t'] * np.sin(prev_theta + u['r1'])
    V[1, 0] = u['t'] * np.cos(prev_theta + u['r1'])
    V[2, 0] = 1
    V[0, 1] = np.cos(prev_theta + u['r1'])
    V[1, 1] = np.sin(prev_theta + u['r1'])
    V[2, 2] = 1

    Rx = V @ R_tilda @ V.T

    R = Fx.T @ Rx @ Fx

    sigma = G @ sigma @ G.T + R

    return mu, sigma


def normalize_angle(phi):
    if phi > np.pi:
        phi = phi - 2 * np.pi
    if phi < -np.pi:
        phi = phi + 2 * np.pi
    return phi
