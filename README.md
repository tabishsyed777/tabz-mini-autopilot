# Mini Autopilot

A small robotics project for learning camera based lane following, rule based planning,
ultrasonic obstacle sensing, data collection, and supervised steering model training. It
includes a virtual car and a live dashboard so the software can be explored without
connecting motors.

## Run the simulator

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py
# open http://localhost:5000
```

You will see a simulated car follow a winding road, stop at a stop sign, wait at a red
light, and brake for an obstacle, while the dashboard shows the scene.

## What it does

| Feature | How it works |
|---|---|
| Lane keeping | Classical OpenCV lane detector, or a supervised CNN trained on recorded steering data |
| Stop signs | Red octagon detection, distance estimated from its size, 3 second stop |
| Traffic lights | Red/green disc detection inside a dark housing; waits for green |
| Obstacle braking | HC-SR04 ultrasonic sensor on a Pi: slows down, then emergency stops |
| Tesla-style dashboard | Perspective road, blue path ribbon (red when braking), signs and obstacles drawn live |
| Data engine | Drive manually while recording, train, redeploy. In autopilot mode the car also saves frames where the net and the lane detector disagree, so you can fix its weak spots |

## Project layout

```
  run.py                 main loop (simulator, camera dry run, or explicitly armed car)
train.py               train the steering network, exports models/autopilot.onnx
autopilot/
  config.py            every tunable number and GPIO pin
  perception.py        lane detector, stop sign + light detector, preprocessing
  planner.py           state machine: drive / stop sign / red light / emergency
  model.py, neural.py  the CNN (PyTorch) and the on-car ONNX runner
  dataset.py           loading, balancing and augmenting data
  recorder.py          saves frames + steering while you drive
  sim.py               built-in driving simulator
  hardware/            Pi camera, L298N motor, servo, ultrasonic drivers
dashboard/             Flask server + the web UI
tools/                 sim_test.py (self-test), make_sim_dataset.py
docs/                  WIRING.md, BUILD_GUIDE.md
```

## The pipeline in one picture

```
camera --> OpenCV lane detector OR trained CNN --> steering --+
    +--> stop sign / traffic light rules ---------------------> planner --> outputs
ultrasonic range sensor -------------------------------------->   |
dashboard (manual controls, mode, recording, E-stop) ---------->   +--> motors (opt-in)
```

## Quick commands

```bash
python tools/make_sim_dataset.py --frames 6000 # make practice training data in the simulator
python -m pip install -r requirements-train.txt
python train.py                                 # train; exports models/autopilot.onnx
python run.py --mode auto                       # simulator with trained model
python run.py --camera usb                      # webcam dry run, motors disabled
python run.py --camera pi                       # Pi camera + sensor, motors disabled
python run.py --camera pi --enable-motors       # explicitly arm physical motors
```

On Raspberry Pi, install the additional hardware drivers from requirements-pi.txt after
the operating system packages are installed. Picamera2 comes from Raspberry Pi OS.

The dashboard binds to localhost by default. To reach it from another device, pass
`--dashboard-host 0.0.0.0`; it has no authentication, so use that only on a trusted,
isolated network. See **docs/BUILD_GUIDE.md** and **docs/WIRING.md** for hardware details.

## Honest limitations

- It is a hobby project, not a safety system. Motors stay disabled unless
  `--enable-motors` is passed. Keep the physical motor power switch within reach and test
  with the wheels off the ground first.
- The built-in detectors use colour and shape, so signs must be printed in saturated red and
  lights need a dark housing. Strong direct sunlight on the track will confuse them.
- The neural net learns steering from examples you provide. Stop sign and traffic light
  handling remain classical colour/shape rules; neither they nor the model are validated
  for public roads.
- Simulator results do not predict real vehicle performance. Tune the camera, steering,
  sensor thresholds, and motor limits for your own track.
