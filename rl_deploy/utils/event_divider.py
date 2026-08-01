# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import time


class EventDivider:
    def __init__(self, context, period_s: float):
        self._context = context
        self._event = context.event
        self._period_s = float(period_s)
        self._deadline = None

    def __call__(self):
        wait_start = time.perf_counter()
        if not self._event.wait(1):
            return False
        self._event.clear()
        now = time.perf_counter()
        if self._deadline is None or now - self._deadline >= self._period_s:
            self._deadline = now
        else:
            self._deadline += self._period_s
            time.sleep(max(0.0, self._deadline - now))

        wait_end = time.perf_counter()
        if hasattr(self._context, 'timing_dict'):
            self._context.timing_dict["dt_divider_wait"] = wait_end - wait_start
            self._context.timing_dict["divider_end"] = wait_end

        return True
