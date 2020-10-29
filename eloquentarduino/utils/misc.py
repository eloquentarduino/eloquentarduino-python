from collections.abc import Iterable
from time import sleep


def is_list(x):
    """
    Check if argument is a list-like object
    :param x:
    :return:
    """
    return isinstance(x, Iterable) and not isinstance(x, str)


def countdown(count):
    """Print a countdown"""
    print('Starting in...', end=' ')
    for i in range(count, 0, -1):
        print('%d...' % i, end='')
        sleep(1)
    print('Go')