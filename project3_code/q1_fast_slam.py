import numpy as np
from FAST.fastslam import main as fast_slam_main

if __name__ == '__main__':
    np.random.seed(0)
    fast_slam_main()