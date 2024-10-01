import functools


def on_keyboard_interrup(on_interrupt_callback):
    def decorator_interrup(func):
        @functools.wraps(func)
        def wrapper_interrup(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                self.__getattribute__(on_interrupt_callback)()

        return wrapper_interrup

    return decorator_interrup
