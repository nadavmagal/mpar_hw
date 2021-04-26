import numpy as np
from tools import normalize_angle


def prediction_step(mu, sigma, u, sigmot):
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
    sigma_rot1, sigma_t, sigma_rot2 = sigmot['sigma_rot1'], sigmot['sigma_t'], sigmot['sigma_rot2']

    m, n = sigma.shape
    prev_theta = mu[2].copy()
    Fx = np.hstack([np.eye(3), np.zeros([3, mu.shape[0] - 3])])
    mu += Fx.T @ np.array([u['t'] * np.cos(prev_theta + u['r1']),
                           u['t'] * np.sin(prev_theta + u['r1']),
                           normalize_angle(u['r1'] + u['r2'])])
    mu[2] = normalize_angle(mu[2])

    # % TODO: Compute the 3x3 Jacobian Gx of the motion model

    # Gx = np.eye(3)
    # Gx[0, 2] = -u['t'] * np.sin(prev_theta + u['r1'])
    # Gx[1, 2] = u['t'] * np.cos(prev_theta + u['r1'])

    # G = np.vstack([np.hstack([Gx, np.zeros([3, n - 3])]), np.hstack([np.zeros([n - 3, 3]), np.eye(n - 3)])])
    Gx = np.hstack([np.zeros([3,2]), np.vstack([-u['t']*np.sin(prev_theta + u['r1']), u['t'] * np.cos(prev_theta + u['r1']), 0])])
    G = np.eye(n) + Fx.T@Gx@Fx

    # % TODO: Compute the 3x3 Jacobian Rx of the motion model
    if True:
        R_tilda = np.diag([sigma_rot1 ** 2, sigma_t ** 2, sigma_rot2 ** 2])
        V = np.zeros([3, 3])
        V[0, 0] = -u['t'] * np.sin(prev_theta + u['r1'])
        V[1, 0] = u['t'] * np.cos(prev_theta + u['r1'])
        V[2, 0] = 1.
        V[0, 1] = np.cos(prev_theta + u['r1'])
        V[1, 1] = np.sin(prev_theta + u['r1'])
        V[2, 2] = 1.

        Rx = V @ R_tilda @ V.T

        R = Fx.T @ Rx @ Fx
    else:
        R3 = np.array([[sigma_rot1, 0, 0],
        [0, sigma_t, 0],
        [0, 0, sigma_rot2 / 10]])
        R = np.zeros_like(sigma)
        R[0: 3, 0: 3] = R3

    sigma = G @ sigma @ G.T + R  # TODO: think about G and R

    return mu, sigma
