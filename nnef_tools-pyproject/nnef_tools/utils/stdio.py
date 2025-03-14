import sys


def set_stdin_to_binary():
    if sys.version_info >= (3, 0):
        sys.stdin = sys.stdin.buffer
    elif sys.platform == 'win32':
        import os, msvcrt
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)


def set_stdout_to_binary():
    if sys.version_info >= (3, 0):
        sys.stdout = sys.stdout.buffer
    elif sys.platform == 'win32':
        import os, msvcrt
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)


def is_stdin_piped():
    return not sys.stdin.isatty()


def is_stdout_piped():
    return not sys.stdout.isatty()
