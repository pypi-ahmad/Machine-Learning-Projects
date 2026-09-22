from pathlib import Path
from subprocess import run
from colors import logcolors


def init():
    run(['git', 'init'], check=True)


def createReadme():
    Path('README.md').touch(exist_ok=True)


def add(filelist):
    for file in filelist:
        # perform git add on file
        print(f"{logcolors.SUCCESS}Adding{logcolors.ENDC}",
              file.split('\\')[-1])
        run(['git', 'add', file], check=True)


# git commit -m "passed message"


def commit(filelist, *args, **kwargs):
    diffarr = kwargs.get('diffarr', -1)
    for file in filelist:
        # ask user for commit message
        msg = str(
            input(
                f'{logcolors.BOLD}Enter the commit message for{logcolors.ENDC} '
                + file.split('\\')[-1] +
                f' {logcolors.BOLD}or enter {logcolors.ERROR}-r{logcolors.ENDC} to reject commit{logcolors.ENDC}'
            ))
        # if msg == -r reject commit
        if (msg == '-r'):
            print(f'{logcolors.ERROR}commit rejected{logcolors.ENDC}')
            if (diffarr != -1):
                diffarr.remove(diffarr[filelist.index(file)])
            filelist.remove(file)
            return False
        # else execute git commit for the file
        # added a comment
        else:
            run(['git', 'commit', '-m', msg], check=True)
            print(
                f'Commited {logcolors.CYAN}{file}{logcolors.ENDC} with msg: {logcolors.BOLD}{msg}{logcolors.ENDC}'
            )


def setremote(url):
    run(['git', 'remote', 'add', 'origin', url], check=True)


def setBranch(branch):
    run(['git', 'branch', '-M', branch], check=True)


# git push


def push(url, branch):
    run(['git', 'push', '-u', url, branch], check=True)
    print(f'{logcolors.SUCCESS}Successfully Pushed Changes{logcolors.ENDC}')
