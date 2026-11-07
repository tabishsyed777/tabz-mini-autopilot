"""Motor + steering control for the real car.

Drive motor : L298N H-bridge, controlled with gpiozero.
Steering    : hobby servo on a PCA9685 board (stable PWM on Pi 5).
"""
import time

from ..config import CFG


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class RealCar:
    def __init__(self):
        from gpiozero import Motor
        from adafruit_servokit import ServoKit

        self.motor = Motor(forward=CFG.pin_motor_in1, backward=CFG.pin_motor_in2,
                           enable=CFG.pin_motor_ena, pwm=True)
        self.kit = ServoKit(channels=16)
        self.servo = self.kit.servo[CFG.servo_channel]
        self.steer = 0.0
        self.throttle = 0.0
        self._last_cmd = time.time()
        self.set(0.0, 0.0)

    def set(self, steer: float, throttle: float):
        """steer: -1 (full left) .. +1 (full right). throttle: -1..+1."""
        steer = _clamp(steer, -1.0, 1.0)
        throttle = _clamp(throttle, -1.0, 1.0)
        self.steer, self.throttle = steer, throttle
        self._last_cmd = time.time()

        s = -steer if CFG.servo_invert else steer
        self.servo.angle = CFG.servo_center_deg + s * CFG.servo_range_deg
        if throttle >= 0:
            self.motor.forward(throttle)
        else:
            self.motor.backward(-throttle)

    def stop(self):
        self.motor.stop()
        self.throttle = 0.0

    def close(self):
        self.stop()
        try:
            self.servo.angle = CFG.servo_center_deg
            self.motor.close()
        except Exception:
            pass
