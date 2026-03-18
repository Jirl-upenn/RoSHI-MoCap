# Metrics (MPJPE cm, JAE °; matched / full)

## Per-dataset breakdown

### dataset1

- GT: 2967 frames, eval window 89.0s, camera frames: 2672

| Method | MPJPE (cm) | JAE (°) | Recall |
| --- | --- | --- | --- |
| IMU-only (naive) | 12.3/ 12.3 | 11.6/ 11.6 | — |
| IMU + EgoAllo root | 12.0/ 12.0 | 11.6/ 11.6 | — |
| EgoAllo | 19.5/ 19.5 | 13.4/ 13.4 | — |
| SAM3D | 10.3/ 10.3 | 10.1/ 10.1 | 100.0% |

### dataset2

- GT: 2546 frames, eval window 76.4s, camera frames: 2292

| Method | MPJPE (cm) | JAE (°) | Recall |
| --- | --- | --- | --- |
| IMU-only (naive) | 14.7/ 14.7 | 12.4/ 12.4 | — |
| IMU + EgoAllo root | 11.8/ 11.8 | 12.4/ 12.4 | — |
| EgoAllo | 20.9/ 20.9 | 15.6/ 15.5 | — |
| SAM3D | 10.6/ 10.6 | 10.4/ 10.4 | 100.0% |

### dataset3

- GT: 1885 frames, eval window 56.6s, camera frames: 1698

| Method | MPJPE (cm) | JAE (°) | Recall |
| --- | --- | --- | --- |
| IMU-only (naive) | 30.3/ 31.7 | 13.3/ 13.5 | — |
| IMU + EgoAllo root | 12.1/ 12.3 | 13.3/ 13.5 | — |
| EgoAllo | 21.7/ 22.2 | 18.3/ 18.0 | — |
| SAM3D | 10.0/ 10.0 | 10.8/ 10.8 | 77.7% |

### dataset4

- GT: 2269 frames, eval window 68.2s, camera frames: 2046

| Method | MPJPE (cm) | JAE (°) | Recall |
| --- | --- | --- | --- |
| IMU-only (naive) | 28.5/ 29.5 | 12.9/ 12.8 | — |
| IMU + EgoAllo root | 12.5/ 12.4 | 12.9/ 12.8 | — |
| EgoAllo | 20.6/ 20.7 | 14.0/ 13.9 | — |
| SAM3D | 10.1/ 10.1 | 10.8/ 10.8 | 91.3% |

### dataset5

- GT: 1890 frames, eval window 56.7s, camera frames: 1702

| Method | MPJPE (cm) | JAE (°) | Recall |
| --- | --- | --- | --- |
| IMU-only (naive) | 23.8/ 39.2 | 12.3/ 12.7 | — |
| IMU + EgoAllo root | 11.4/ 13.4 | 12.3/ 12.7 | — |
| EgoAllo | 13.1/ 14.4 | 12.7/ 13.1 | — |
| SAM3D | 9.8/ 9.8 | 10.7/ 10.7 | 54.8% |

### dataset6

- GT: 1418 frames, eval window 42.6s, camera frames: 1278

| Method | MPJPE (cm) | JAE (°) | Recall |
| --- | --- | --- | --- |
| IMU-only (naive) | 25.4/ 26.5 | 12.0/ 12.2 | — |
| IMU + EgoAllo root | 12.1/ 13.2 | 12.0/ 12.2 | — |
| EgoAllo | 18.3/ 20.1 | 14.1/ 14.2 | — |
| SAM3D | 10.7/ 10.7 | 10.8/ 10.8 | 70.7% |

### dataset7

- GT: 1466 frames, eval window 44.0s, camera frames: 1320

| Method | MPJPE (cm) | JAE (°) | Recall |
| --- | --- | --- | --- |
| IMU-only (naive) | 22.7/ 22.7 | 11.8/ 11.8 | — |
| IMU + EgoAllo root | 11.0/ 11.0 | 11.8/ 11.8 | — |
| EgoAllo | 15.4/ 15.4 | 13.3/ 13.3 | — |
| SAM3D | 10.6/ 10.6 | 10.6/ 10.6 | 100.0% |

### dataset8

- GT: 1514 frames, eval window 45.4s, camera frames: 1364

| Method | MPJPE (cm) | JAE (°) | Recall |
| --- | --- | --- | --- |
| IMU-only (naive) | 15.7/ 16.8 | 11.6/ 11.6 | — |
| IMU + EgoAllo root | 10.3/ 10.6 | 11.6/ 11.6 | — |
| EgoAllo | 18.0/ 18.2 | 17.6/ 17.6 | — |
| SAM3D | 11.0/ 11.0 | 10.1/ 10.1 | 87.8% |

## Per-session averages

### Session 1 (dataset1-4)

| Method | MPJPE (cm) | JAE (°) | Recall |
| --- | --- | --- | --- |
| IMU-only (naive) | 21.5/ 22.0 | 12.6/ 12.6 | — |
| IMU + EgoAllo root | 12.1/ 12.1 | 12.6/ 12.6 | — |
| EgoAllo | 20.7/ 20.8 | 15.3/ 15.2 | — |
| SAM3D | 10.3/ 10.3 | 10.5/ 10.5 | 92.2% |

### Session 2 (dataset5-8)

| Method | MPJPE (cm) | JAE (°) | Recall |
| --- | --- | --- | --- |
| IMU-only (naive) | 21.9/ 26.3 | 11.9/ 12.1 | — |
| IMU + EgoAllo root | 11.2/ 12.1 | 11.9/ 12.1 | — |
| EgoAllo | 16.2/ 17.0 | 14.4/ 14.5 | — |
| SAM3D | 10.5/ 10.5 | 10.6/ 10.6 | 78.3% |
