# Hardware Examples

This directory contains standalone examples for interfacing with the RoSHI hardware.

## IMU Reader (`imu_reader.py`)

A unified reader, recorder, and visualizer for up to 9 BNO085 IMUs connected to a single ESP32 serial receiver streaming multiplexed packets.

### Hardware Setup

- **MCU**: ESP32-S3 Feather (Adafruit)
- **IMU**: BNO085 x9, connected via multiplexed I2C
- **Protocol**: Serial USB at 115200 baud
- **Packet format**: `ID:<id> | r:<roll> p:<pitch> y:<yaw> | ax:... ay:... az:... | gx:... gy:... gz:... | qI:... qJ:... qK:... qW:... | batt:<pct>%`

### Usage

```bash
# Visualize IMU orientations in a 3x3 matplotlib grid
python imu_reader.py --mode visualize

# Record IMU data to CSV
python imu_reader.py --mode record --duration 60

# Specify serial port (default: /dev/ttyACM0, use 'auto' for auto-detect)
python imu_reader.py --port auto
```

### Dependencies

```
pyserial
numpy
matplotlib
```

### Integration

The `IMUReader` and `IMUDataRecorder` classes can be imported and used in other scripts:

```python
from hardware.imu_reader import IMUReader, IMUDataRecorder

recorder = IMUDataRecorder(num_imus=9)
recorder.connect(port="auto")
recorder.start_recording()

# In your loop:
readings = recorder.read_and_save_data()
snapshot = recorder.get_latest_snapshot()
```

For the iOS companion app, see [RoSHI-App](https://github.com/Jirl-upenn/RoSHI-App).
