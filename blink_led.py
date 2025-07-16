import time
import os

LED_PATH = '/sys/class/leds/beaglebone:green:usr0'

while True:
    with open(LED_PATH + '/brightness', 'w') as f:
        f.write('1')  # Turn the LED on
    time.sleep(0.5)
    with open(LED_PATH + '/brightness', 'w') as f:
        f.write('0')  # Turn the LED off
    time.sleep(0.5)
