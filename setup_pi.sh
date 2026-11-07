#!/usr/bin/env bash
# One-shot setup for Raspberry Pi OS (64-bit, Bookworm). Run:  bash setup_pi.sh
set -e
sudo apt update
sudo apt install -y python3-venv python3-opencv python3-picamera2 python3-numpy i2c-tools
sudo raspi-config nonint do_i2c 0          # enable I2C for the PCA9685 servo board
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install flask onnxruntime waitress gpiozero lgpio adafruit-circuitpython-servokit
echo
echo "Done. Next:  source .venv/bin/activate && python run.py --camera pi --no-motors"
echo "Check the servo board is seen:  i2cdetect -y 1   (expect address 40)"
