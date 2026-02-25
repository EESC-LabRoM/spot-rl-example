# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import time


class EventDivider:
    def __init__(self, context, factor: int):
        self._context = context
        self._event = context.event
        self._factor = factor

    def __call__(self):
        count = 0
        wait_start = time.perf_counter()

        while count < self._factor:
            if not self._event.wait(1):
                return False

            count += 1
            self._event.clear()
            time.sleep(0.001)

        wait_end = time.perf_counter()
        if hasattr(self._context, 'timing_dict'):
            self._context.timing_dict["dt_divider_wait"] = wait_end - wait_start
            self._context.timing_dict["divider_end"] = wait_end

        return True
