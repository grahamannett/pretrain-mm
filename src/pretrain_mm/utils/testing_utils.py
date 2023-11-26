import time

from pretrain_mm import logger


class TimerMixin:
    """use within unittest.TestCase to time each test method"""

    _time_after_test: bool = False
    _timers = {}

    def setUp(self):
        self.timer_setup()
        super().setUp()

    def tearDown(self):
        if self._time_after_test:
            self.check_timer(self._testMethodName)

        super().tearDown()

    def timer_setup(self, name: str = None):
        name = name or self._testMethodName
        self._running_timer = name
        logger.info(f"[yellow]FUNC:{self._testMethodName}")
        self._timers[name] = time.perf_counter()

    def check_timer(self, name: str = None, extra_print: str = None):
        time_now = time.perf_counter()
        name = name or self._running_timer
        elapsed_time = time_now - self._timers[name]

        start_str = f"[red]⏰[{name}] " if name else "[red] ⏰"
        if extra_print:
            start_str += extra_print
        logger.info(f"{start_str}: {elapsed_time} seconds")

        # reset timer
        self._timers[name] = time.perf_counter()
