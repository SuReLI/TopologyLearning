import time
from copy import deepcopy
from datetime import datetime




class Stopwatch:
    def __init__(self):
        temp_time = datetime.now()
        self.duration = temp_time - temp_time  # Initialise to a duration of 0.
        self.last_start = None

    def start(self):
        assert not self.is_running(), "Cannot start a running stopwatch."
        self.last_start = datetime.now()

    def stop(self):
        assert self.is_running(), "Stopwatch has not been started. There is probably an error."
        self.duration += datetime.now() - self.last_start
        self.last_start = None
        return self.duration

    def get_duration(self):
        if self.is_running():
            return self.duration + (datetime.now() - self.last_start)
        else:
            return deepcopy(self.duration)

    def is_running(self):
        return self.last_start is not None

    def reset(self, start=False):
        self.__init__()
        if start:
            self.start()


if __name__ == "__main__":
    s = Stopwatch()
    s.start()
    time.sleep(2)
    duration = s.get_duration()
    print("duration = ", duration.seconds + duration.microseconds * 1e-6, sep="")
    time.sleep(2)
    duration = s.get_duration()
    print("duration = ", duration.seconds + duration.microseconds * 1e-6, sep="")
    s.reset(start=True)
    print("clock reset")
    time.sleep(2)
    duration = s.get_duration()
    print("duration = ", duration.seconds + duration.microseconds * 1e-6, sep="")
    time.sleep(2)
    duration = s.get_duration()
    print("duration = ", duration.seconds + duration.microseconds * 1e-6, sep="")
    s.stop()
    duration = s.get_duration()
    print("duration = ", duration.seconds + duration.microseconds * 1e-6, sep="")

