# Wiring

Pin numbers below are **BCM GPIO numbers** (the ones in `autopilot/config.py`), with the
physical header pin in brackets.

## Power: read this first

- **Raspberry Pi 5**: its own USB-C power bank (5 V, ideally 5 A capable).
- **Drive motor + steering servo**: a separate battery (e.g. 2S LiPo 7.4 V or 6 x AA).
- **All grounds must be connected together** (Pi GND, L298N GND, PCA9685 GND, battery GND).
- Never power the motor or servo from the Pi's 5 V pin.
- Fit a physical on/off switch on the motor battery. It is your last-resort emergency stop.

## L298N motor driver (drive motor)

| L298N | Connects to |
|---|---|
| IN1 | GPIO 23 (pin 16) |
| IN2 | GPIO 24 (pin 18) |
| ENA | GPIO 25 (pin 22). **Remove the ENA jumper cap first** so PWM controls speed |
| +12V (VS) | Motor battery + |
| GND | Motor battery - **and** Pi GND |
| OUT1 / OUT2 | Drive motor |

The L298N drops about 2 V, so a 7.4 V pack gives the motor roughly 5 V. That is fine for
a small gear motor. If your car has two drive motors, wire them in parallel to OUT1/OUT2.

## PCA9685 servo board (steering)

| PCA9685 | Connects to |
|---|---|
| VCC | 3.3 V (pin 1) |
| SDA | GPIO 2 (pin 3) |
| SCL | GPIO 3 (pin 5) |
| GND | GND (pin 6) |
| V+ | 5-6 V servo supply (from a BEC or buck converter, **not** the Pi) |
| Channel 0 | Steering servo (signal, V+, GND) |

Enable I2C (`setup_pi.sh` does it) and check with `i2cdetect -y 1`. You should see `40`.

## HC-SR04 ultrasonic sensor

The sensor runs at 5 V but the Pi's GPIO is **3.3 V only**. The ECHO pin needs a voltage divider.

| HC-SR04 | Connects to |
|---|---|
| VCC | 5 V (pin 2) |
| GND | GND |
| TRIG | GPIO 5 (pin 29) |
| ECHO | 1 kΩ resistor, then GPIO 6 (pin 31). Also a 2 kΩ resistor from that junction to GND |

```
ECHO ---[1k]---+--- GPIO 6
               |
              [2k]
               |
              GND
```

## Camera

Pi Camera Module (v2 / v3) into the CSI port using the cable that fits the Pi 5's small
connector. Mount it about 10-15 cm above the ground, tilted slightly down, centred on the car.
