"""HC-SR04 distance sensor. Returns centimetres (999 if nothing in range)."""
from ..config import CFG


class RealUltrasonic:
    def __init__(self):
        from gpiozero import DistanceSensor
        # max_distance in metres. queue_len smooths out noisy readings.
        self.sensor = DistanceSensor(echo=CFG.pin_us_echo, trigger=CFG.pin_us_trigger,
                                     max_distance=2.0, queue_len=3)

    def read_cm(self) -> float:
        d = self.sensor.distance * 100.0
        return 999.0 if d >= 199.0 else d

    def close(self):
        self.sensor.close()
