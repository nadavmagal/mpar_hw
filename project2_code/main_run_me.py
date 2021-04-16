import os
import time
from common import load_data
from q1_kalman_filter import kalman_filter
from q2_extended_kalman_filter import extended_kalman_filter

def main():
    basedir = '/home/nadav/studies/mapping_and_perception_autonomous_robots/kitti_data/orginaized_data'
    date = '2011_09_30'
    dataset_number = '0033'

    result_dir = r'/home/nadav/studies/mapping_and_perception_autonomous_robots/project_2/results'
    cur_date_time = time.strftime("%Y.%m.%d-%H.%M")

    result_dir_timed = os.path.join(result_dir, f'{cur_date_time}')
    print(f'saving to: {result_dir_timed}')
    os.makedirs(result_dir_timed, exist_ok=True)

    data = load_data(basedir, date, dataset_number)

    # Q1
    kalman_filter(result_dir_timed, data)
    # Q2
    extended_kalman_filter(result_dir_timed, data)


if __name__ == "__main__":
    main()
