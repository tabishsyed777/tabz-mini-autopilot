"""Wraps SimWorld so it looks like the real camera / car / ultrasonic."""
import time

from ..sim import SimWorld


class SimRig:
    """One object that acts as camera + car + ultrasonic in simulation."""

    def __init__(self, seed=0, with_events=True, realtime=True):
        self.world = SimWorld(seed, with_events)
        self.realtime = realtime
        self._t = time.time()
        self.steer = 0.0
        self.throttle = 0.0

    # camera
    def read(self):
        now = time.time()
        dt = min(0.1, now - self._t) if self.realtime else 1 / 30
        self._t = now
        self.world.step(self.steer, self.throttle, dt)
        self.world.tick_lights(dt)
        if self.world.crashed:
            self.world.reset_to_road()
        return self.world.render()

    # car
    def set(self, steer, throttle):
        self.steer, self.throttle = steer, throttle

    def stop(self):
        self.throttle = 0.0

    # ultrasonic
    def read_cm(self):
        return self.world.ultrasonic_cm()

    def close(self):
        pass
