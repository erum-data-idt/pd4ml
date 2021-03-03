import sys
import pkg_resources
import platform
import getpass


def meta(self):
    return {
        "username": getpass.getuser(),
        "hostname": platform.node(),
    }


def runtime(self):
    pyv = sys.version_info
    return {
        "python": ".".join(map(str, [pyv.major, pyv.minor, pyv.micro])),
        "OS": f"{platform.system()} - {platform.release()}",
    }


def env(self):
    return {pkg.key: pkg.version for pkg in pkg_resources.working_set}
