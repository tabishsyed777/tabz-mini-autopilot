# Build guide

## Parts

| Part | Notes |
|---|---|
| Raspberry Pi 5 (4 GB or 8 GB) | A Pi 4 also works; the net is small |
| Pi Camera Module 3 (or v2) | USB webcam also works with `--camera usb` |
| 1/10 RC car chassis with servo steering | Needs a steering servo and one drive motor. Ackermann steering gives the most car-like result |
| L298N motor driver | |
| PCA9685 16-channel servo board | Gives jitter-free servo PWM on the Pi 5 |
| HC-SR04 ultrasonic sensor | Plus a 1 kΩ and a 2 kΩ resistor |
| 2S LiPo (or 6 x AA) for motors | Separate from the Pi |
| Power bank for the Pi | 5 V / 3 A minimum, 5 A recommended for the Pi 5 |
| Jumper wires, small breadboard, switch | |
| White tape (2-3 cm wide), dark floor or mat | For the track |
| Printed stop sign + traffic light | See "Make the track" |

Cost depends heavily on what you already own and where you buy.

## 1. Set up the Raspberry Pi

1. Flash **Raspberry Pi OS 64-bit** with Raspberry Pi Imager. Set Wi-Fi and enable SSH in the imager.
2. SSH in, copy this project over (`scp -r mini-autopilot pi@raspberrypi.local:~`), then:
   ```bash
   cd mini-autopilot
   bash setup_pi.sh
   source .venv/bin/activate
   python -m pip install -r requirements-pi.txt
   ```
3. Dry run with the wheels disabled:
   ```bash
   python run.py --camera pi --no-motors
   ```
   The dashboard binds to localhost by default. For access from your phone on a trusted
   local network, start with `--dashboard-host 0.0.0.0` and open the Pi dashboard URL.
   The dashboard has no authentication.

## 2. Build and wire the car

Mount the Pi, camera, L298N, PCA9685 and sensor on the chassis, then wire following
`docs/WIRING.md`. **Raise the car on a stand so the wheels spin freely** for the next steps.

## 3. Calibrate steering and motor

Run `python run.py --camera pi --enable-motors`, switch to Manual, and use the joystick
or arrow keys. Keep the car on a stand and the physical power switch within reach.

- Steering goes the wrong way: set `servo_invert = True` in `config.py`.
- Wheels hit their limit or turn too little: change `servo_range_deg`.
- Wheels not straight at zero: change `servo_center_deg` (try 85 to 95).
- Motor spins backwards: swap the two motor wires, or swap `pin_motor_in1` / `pin_motor_in2`.

## 4. Make the track

- **Lane**: two parallel white tape lines on a dark floor, about 3 x the car's width apart.
  Make some gentle curves. If you use black tape on a light floor, set `line_is_bright = False`.
- **Stop sign**: print a saturated red octagon, glue to card, mount on a stick beside the lane.
  Measure its height and set `sign_height_m` in `config.py`.
- **Traffic light**: a **black card** with a red disc and a green disc you can swap (or LEDs).
  The black card matters: the detector recognises a light by its dark housing.
  Set `light_diam_m` to the disc diameter.
- **Obstacle**: any orange box or cup in the lane.
- **Camera focal length**: set `focal_px = (320 / 2) / tan(hfov / 2)`. Pi Camera 3 standard
  lens (66 degrees) is about 246.

## 5. First autonomous run (no training needed)

Switch to **Lane detector**. It steers using OpenCV only. Start slow (`cruise_speed = 0.25`).
Place the car on the track. Press **Space** or the **Emergency stop** button at any time.

Check the camera panel: the green centre line should sit between the blue lane lines.
If lines are missed, adjust lighting, or the threshold in `LaneDetector._mask`.

## 6. Teach it to drive (the neural net)

1. In **Manual** mode press **Record driving**. Drive 10 to 15 laps, **both directions**.
2. Also record *recoveries*: drift toward a line on purpose and steer back. The net needs to see how to fix mistakes.
3. Vary the lighting (day, lamp, shadows). Aim for 5,000 to 10,000 frames.
4. Copy the `data/` folder to a laptop (`scp -r pi@raspberrypi.local:~/mini-autopilot/data .`).
5. On the laptop:
   ```bash
   pip install -r requirements-train.txt
   python train.py
   ```
   Watch the "steering error" column fall; it is the average miss in steering units (full range is -1 to +1).
6. Copy `models/autopilot.onnx` to the car's `models/` folder.
7. Run `python run.py --camera pi --mode auto` or press **Neural net** in the dashboard.

No time to collect data yet? `python tools/make_sim_dataset.py` plus `python train.py` lets you
test the whole training pipeline in the simulator first. A model trained only on the
simulator will not drive a real car.

## 7. The data engine (making it better)

In autopilot mode, press **Record driving**. The car now saves frames where the neural net and the lane detector
strongly disagree into `data/auto_*`. Look through them, delete wrong ones, retrain, redeploy.
This is a tiny version of the loop real fleets use: drive, collect hard cases, retrain, ship.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: adafruit_servokit` or Blinka errors on Pi 5 | `pip install adafruit-blinka rpi-lgpio` |
| Servo jitters | Use a separate 5-6 V supply on PCA9685 V+; check grounds are joined |
| Car wobbles left and right | Lower `steer_smoothing` toward 0.4 or reduce speed; check `servo_range_deg` |
| Stop sign never triggers | Check `sign_height_m` and `focal_px`; make the red more saturated |
| Obstacle reading jumps around | Check the ECHO voltage divider and keep the sensor level |
| Dashboard shows no camera | Run `rpicam-hello` to confirm the camera works at all |

## Ideas to extend it

- Add a second camera or a wide-angle lens for more context.
- Predict speed as well as steering (add a second output in `model.py`).
- Run the net on a Coral/Hailo accelerator for higher frame rates.
- Record a demo video of the dashboard next to the real car; it is what makes a project shareable.
