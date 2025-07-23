import gpiod
import time
gpio14 = gpiod.find_line('GPIO14')
gpio14.request(consumer='beagle', type=gpiod.LINE_REQ_DIR_OUT, default_val=0)

while True:
    gpio14.set_value(1)
    time.sleep(1)
    gpio14.set_value(0)
    time.sleep(1)