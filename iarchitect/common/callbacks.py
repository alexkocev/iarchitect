from IPython.core.display import clear_output, display


def output_updater(*args):
    def update(step,trainer):
        clear_output(wait = True)
        for a in args:
            display(a)
    return update