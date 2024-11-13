class LinearDiscountScheduler:
    def __init__(self, initial_value, final_value, total_steps):
        self.initial_value = initial_value
        self.final_value = final_value
        self.total_steps = total_steps
        self.current_step = 0

    def __call__(self):
        if self.current_step >= self.total_steps:
            return self.final_value
        self.current_step += 1

        # Linear interpolation between initial and final value
        return self.initial_value + (self.final_value - self.initial_value) * (
            self.current_step / self.total_steps
        )


class ExponentialDiscountScheduler:
    def __init__(self, initial_value, final_value, total_steps):
        self.initial_value = initial_value
        self.final_value = final_value
        self.total_steps = total_steps
        self.current_step = 0

    def __call__(self):
        if self.current_step >= self.total_steps:
            return self.final_value
        self.current_step += 1

        # Exponential interpolation between initial and final value
        return self.initial_value * (
            self.final_value / self.initial_value
        ) ** (self.current_step / self.total_steps)
