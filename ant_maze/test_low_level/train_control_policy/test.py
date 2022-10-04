from utils.stopwatch import Stopwatch
from time import sleep

stopwatch = Stopwatch()
stopwatch.start()
for i in range(5):
    sleep(1)
    print(stopwatch.get_duration())
