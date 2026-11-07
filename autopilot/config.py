"""All tunable settings in one place. Edit this file to match your build."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # ---- Camera -------------------------------------------------------
    cam_width: int = 320
    cam_height: int = 240
    cam_fps: int = 30

    # ---- Neural network input (what the model actually sees) ----------
    # The top part of the image is sky/background. We crop it away and
    # shrink the rest. Smaller input = faster on a Raspberry Pi.
    crop_top_frac: float = 0.40      # remove top 40% of the frame
    net_width: int = 128
    net_height: int = 48

    # ---- Camera calibration + real-world sizes (used to estimate distance) ----
    # focal_px = (cam_width / 2) / tan(horizontal_fov / 2).  Pi Camera v3 (66 deg): ~246.
    focal_px: float = 210.0
    sign_height_m: float = 0.21      # height of YOUR printed stop sign (measure it)
    light_diam_m: float = 0.10       # diameter of YOUR red/green light disc

    # ---- Lane marking colour on your track ----------------------------
    line_is_bright: bool = True      # True = white tape on dark floor, False = black tape on light floor

    # ---- Driving ------------------------------------------------------
    cruise_speed: float = 0.35       # 0..1 motor power while driving
    slow_speed: float = 0.22         # used in tight curves / near objects
    steer_smoothing: float = 0.55    # 0 = no smoothing, closer to 1 = very smooth

    # ---- Safety -------------------------------------------------------
    brake_distance_cm: float = 25.0  # emergency stop if obstacle closer
    slow_distance_cm: float = 55.0   # slow down if obstacle closer
    stop_sign_pause_s: float = 3.0

    # ---- Hardware pins (BCM numbering) --------------------------------
    pin_motor_in1: int = 23          # L298N IN1
    pin_motor_in2: int = 24          # L298N IN2
    pin_motor_ena: int = 25          # L298N ENA (PWM speed)
    pin_us_trigger: int = 5          # HC-SR04 trigger
    pin_us_echo: int = 6             # HC-SR04 echo (through voltage divider!)
    servo_channel: int = 0           # PCA9685 channel for steering servo
    servo_center_deg: float = 90.0
    servo_range_deg: float = 28.0    # +/- degrees from center (tune to your car)
    servo_invert: bool = False       # flip if car steers the wrong way

    # ---- Paths --------------------------------------------------------
    data_dir: Path = ROOT / "data"
    model_path: Path = ROOT / "models" / "autopilot.onnx"

    # ---- Dashboard ----------------------------------------------------
    # Keep the unauthenticated control panel local by default.
    dash_host: str = "127.0.0.1"
    dash_port: int = 5000


CFG = Config()
