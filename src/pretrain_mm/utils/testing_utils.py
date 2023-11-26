import time

from pretrain_mm import logger


class TimerMixin:
    """use within unittest.TestCase to time each test method"""

    _time_after_test: bool = True

    def timer_setup(self):
        self.start_time = time.perf_counter()
        logger.info(f"[yellow]FUNC:{self._testMethodName}")

    def check_timer(self, name: str = None):
        time_now = time.perf_counter()
        elapsed_time = time_now - self.start_time
        start_str = f"[red]⏰[{name}] " if name else "[red] ⏰"
        logger.info(f"{start_str}: {elapsed_time} seconds")

        # reset timer
        self.start_time = time.perf_counter()

    def setUp(self):
        self.timer_setup()
        super().setUp()

    def tearDown(self):
        if self._time_after_test:
            self.check_timer(self._testMethodName)

        super().tearDown()
