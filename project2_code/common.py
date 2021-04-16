import pykitti
import numpy as np

def load_data(basedir, date, dataset_number):
    data = pykitti.raw(basedir, date, dataset_number)
    return data

def exract_data_vectors(data):
    oxts = data.oxts
    delta_time = np.append([0], np.array(
        [cur.total_seconds() for cur in np.array(data.timestamps[1::]) - np.array(data.timestamps[:-1:])]))
    time_sec = np.cumsum(delta_time)

    point_imu = np.array([0, 0, 0, 1])
    car_w_coordinates_m = []
    car_w_coordinates_lon_lat = []
    car_yaw = []
    car_vf = []
    car_wz = []
    for cur_oxt in oxts:
        car_w_coordinates_m.append(cur_oxt.T_w_imu.dot(point_imu)[0:2])
        car_w_coordinates_lon_lat.append([cur_oxt.packet.lon, cur_oxt.packet.lat])
        car_yaw.append(cur_oxt.packet.yaw)
        car_vf.append(cur_oxt.packet.vf)
        car_wz.append(cur_oxt.packet.wz)
    car_w_coordinates_m = np.array(car_w_coordinates_m)
    car_w_coordinates_lon_lat = np.array(car_w_coordinates_lon_lat)
    car_yaw = np.array(car_yaw)
    car_vf = np.array(car_vf)
    car_wz = np.array(car_wz)
    return car_w_coordinates_m, car_w_coordinates_lon_lat, car_yaw, car_vf, car_wz, delta_time, time_sec
