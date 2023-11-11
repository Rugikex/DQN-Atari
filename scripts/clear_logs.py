import shutil


def clear_logs():
    """
    Clears all logs
    """
    shutil.rmtree('logs')
    print("Logs cleared")


if __name__ == '__main__':
    clear_logs()
