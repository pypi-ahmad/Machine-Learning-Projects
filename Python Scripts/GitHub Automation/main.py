import argparse

import repoInfo
from filechange import ischanged
from colors import logcolors
import pyfiglet
import logger
from utils import initCommands


def init(poll_interval: float) -> None:
    info = repoInfo.checkinfoInDir()
    url, branch = info
    logger.checkdata(url, branch)
    if ('n' in info):
        initCommands(info)
    else:
        print(
            f'{logcolors.BOLD}Retrieving info from git directory{logcolors.ENDC}'
        )
        print(
            f'{logcolors.CYAN}URL:{logcolors.ENDC} {url} , {logcolors.CYAN}Branch:{logcolors.ENDC} {branch}'
        )
        ischanged(url, branch, poll_interval=poll_interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monitor a Git repository and interactively commit changes.')
    parser.add_argument('--interval', type=float, default=1.0, help='Seconds between change scans')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be greater than zero')
    f = pyfiglet.figlet_format('G - AUTO', font='5lineoblique')
    print(f"{logcolors.BOLD}{f}{logcolors.ENDC}")
    init(args.interval)
import argparse
