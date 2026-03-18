# Metrics by Activity (MPJPE cm, JAE deg)

### walk_march_jog_run

- Eval: 36.0s, camera frames: 1081


| Method             | MPJPE (cm) | JAE (deg) | Recall |
| ------------------ | ---------- | --------- | ------ |
| IMU-only (naive)   | 9.6        | 11.5      | —      |
| IMU + EgoAllo root | 12.0       | 11.2      | —      |
| EgoAllo            | 10.9       | 14.5      | —      |
| IMU-TTO (Ours)     | 11.6       | 11.8      | —      |
| SAM3D              | 9.9        | 10.4      | 100.0% |


### stretch_boxing_bow_wave

- Eval: 99.0s, camera frames: 2972


| Method             | MPJPE (cm) | JAE (deg) | Recall |
| ------------------ | ---------- | --------- | ------ |
| IMU-only (naive)   | 15.7       | 12.0      | —      |
| IMU + EgoAllo root | 11.8       | 11.8      | —      |
| EgoAllo            | 8.9        | 13.5      | —      |
| IMU-TTO (Ours)     | 8.2        | 11.5      | —      |
| SAM3D              | 10.6       | 9.9       | 100.0% |


### jumping-jack_squat_one-leg-squad

- Eval: 61.8s, camera frames: 1854


| Method             | MPJPE (cm) | JAE (deg) | Recall |
| ------------------ | ---------- | --------- | ------ |
| IMU-only (naive)   | 14.8       | 13.0      | —      |
| IMU + EgoAllo root | 11.6       | 13.2      | —      |
| EgoAllo            | 11.7       | 18.0      | —      |
| IMU-TTO (Ours)     | 8.4        | 11.6      | —      |
| SAM3D              | 10.1       | 11.2      | 100.0% |


### pick-up-box

- Eval: 49.0s, camera frames: 1471


| Method             | MPJPE (cm) | JAE (deg) | Recall |
| ------------------ | ---------- | --------- | ------ |
| IMU-only (naive)   | 26.7       | 13.8      | —      |
| IMU + EgoAllo root | 15.4       | 13.9      | —      |
| EgoAllo            | 10.7       | 16.3      | —      |
| IMU-TTO (Ours)     | 10.3       | 13.0      | —      |
| SAM3D              | 10.7       | 10.5      | 99.9%  |


### walk-sayhi-walk

- Eval: 74.3s, camera frames: 2230


| Method             | MPJPE (cm) | JAE (deg) | Recall |
| ------------------ | ---------- | --------- | ------ |
| IMU-only (naive)   | 18.1       | 12.4      | —      |
| IMU + EgoAllo root | 11.0       | 12.5      | —      |
| EgoAllo            | 9.3        | 12.9      | —      |
| IMU-TTO (Ours)     | 9.0        | 10.4      | —      |
| SAM3D              | 9.7        | 10.9      | 77.0%  |


### pickup-walkaround

- Eval: 53.7s, camera frames: 1612


| Method             | MPJPE (cm) | JAE (deg) | Recall |
| ------------------ | ---------- | --------- | ------ |
| IMU-only (naive)   | 27.3       | 12.6      | —      |
| IMU + EgoAllo root | 14.7       | 12.6      | —      |
| EgoAllo            | 11.1       | 13.7      | —      |
| IMU-TTO (Ours)     | 11.3       | 11.1      | —      |
| SAM3D              | 11.1       | 11.1      | 50.0%  |


### walk/jog-back-and-forth

- Eval: 41.1s, camera frames: 1234


| Method             | MPJPE (cm) | JAE (deg) | Recall |
| ------------------ | ---------- | --------- | ------ |
| IMU-only (naive)   | 14.4       | 11.9      | —      |
| IMU + EgoAllo root | 10.1       | 11.8      | —      |
| EgoAllo            | 8.4        | 13.0      | —      |
| IMU-TTO (Ours)     | 9.2        | 10.6      | —      |
| SAM3D              | 10.2       | 10.6      | 100.0% |


### jump-around

- Eval: 48.7s, camera frames: 1464


| Method             | MPJPE (cm) | JAE (deg) | Recall |
| ------------------ | ---------- | --------- | ------ |
| IMU-only (naive)   | 15.2       | 11.8      | —      |
| IMU + EgoAllo root | 11.7       | 11.7      | —      |
| EgoAllo            | 11.3       | 16.7      | —      |
| IMU-TTO (Ours)     | 10.3       | 11.8      | —      |
| SAM3D              | 10.9       | 10.3      | 90.4%  |


### sliding

- Eval: 46.8s, camera frames: 1405


| Method             | MPJPE (cm) | JAE (deg) | Recall |
| ------------------ | ---------- | --------- | ------ |
| IMU-only (naive)   | 9.2        | 9.2       | —      |
| IMU + EgoAllo root | 11.4       | 9.0       | —      |
| EgoAllo            | 9.8        | 17.6      | —      |
| IMU-TTO (Ours)     | 9.1        | 14.6      | —      |
| SAM3D              | 18.6       | 11.1      | 98.1%  |


### tennis

- Eval: 54.0s, camera frames: 1621


| Method             | MPJPE (cm) | JAE (deg) | Recall |
| ------------------ | ---------- | --------- | ------ |
| IMU-only (naive)   | 23.0       | 9.0       | —      |
| IMU + EgoAllo root | 14.5       | 8.8       | —      |
| EgoAllo            | 15.5       | 19.6      | —      |
| IMU-TTO (Ours)     | 12.9       | 17.2      | —      |
| SAM3D              | 21.9       | 11.3      | 100.0% |


### ball-throwing-catching

- Eval: 45.4s, camera frames: 1362


| Method             | MPJPE (cm) | JAE (deg) | Recall |
| ------------------ | ---------- | --------- | ------ |
| IMU-only (naive)   | 16.0       | 8.4       | —      |
| IMU + EgoAllo root | 11.7       | 8.4       | —      |
| EgoAllo            | 9.8        | 15.4      | —      |
| IMU-TTO (Ours)     | 8.9        | 15.2      | —      |
| SAM3D              | 24.3       | 11.1      | 100.0% |


## Overall Average


| Method             | MPJPE (cm) | JAE (deg) | Recall |
| ------------------ | ---------- | --------- | ------ |
| IMU-only (naive)   | 17.3       | 11.4      | —      |
| IMU + EgoAllo root | 12.3       | 11.4      | —      |
| EgoAllo            | 10.7       | 15.6      | —      |
| IMU-TTO (Ours)     | 9.9        | 12.6      | —      |
| SAM3D              | 13.5       | 10.8      | 92.3%  |


## Dataset Summary


|                    | Dataset 1  |           | Dataset 2  |           | Dataset 3  |           |
| ------------------ | ---------- | --------- | ---------- | --------- | ---------- | --------- |
|                    | MPJPE (cm) | JAE (deg) | MPJPE (cm) | JAE (deg) | MPJPE (cm) | JAE (deg) |
| IMU-only (naive)   | 16.7       | 12.6      | 18.8       | 12.2      | 16.1       | 8.9       |
| IMU + EgoAllo root | 12.7       | 12.5      | 11.9       | 12.2      | 12.5       | 8.7       |
| EgoAllo            | 10.6       | 15.6      | 10.0       | 14.1      | 11.7       | 17.5      |
| IMU-TTO (Ours)     | 9.6        | 12.0      | 9.9        | 11.0      | 10.3       | 15.6      |
| SAM3D              | 10.3       | 10.5      | 10.5       | 10.7      | 21.6       | 11.2      |


