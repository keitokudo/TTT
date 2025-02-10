from collections import namedtuple
import subprocess
import os


def argmax(a):
    return a.index(max(a))

def argmin(a):
    return a.index(min(a))


def dict_to_namedtuple(cls_name, data):
    cls =  namedtuple(
        cls_name.replace("_", "").title().replace(" ", ""),
        data.keys()
    )
    
    return cls(
        *(
            dict_to_namedtuple(
                cls_name + '_' + k,
                v
            ) if isinstance(v, dict) else v for k, v in data.items()
        )
    )


def _exec_command(command_list):
    proc = subprocess.Popen(
        command_list,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ,
    )
    
    while True:
        line = proc.stdout.readline()

        if proc.returncode is not None:
            if proc.returncode:
                raise Exception(f"{' '.join(command_list)} failed...")
        
        if line:
            yield line.decode()

        if not line and proc.poll() is not None:
            break


def exec_command(command_list):
    for stdout_line in _exec_command(command_list):
        print(stdout_line, end="")

def exec_command_to_string(command_list):
    return "".join(_exec_command(command_list))

def number_of_lines(path):
    return int(
        exec_command_to_string(
            ["wc", "-l", str(path)]
        ).split(" ")[0]
    )
