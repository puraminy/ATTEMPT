import logging
import os
from os.path import expanduser
from pytz import timezone
import datetime
from pathlib import Path
main_args = {}

def args(key, default="no_default"):
    if key in main_args:
        return main_args[key]
    else:
        return default

def get_full_tag(as_str=False):
    return get_tag(main_args["full_tag"], main_args, as_str)

def get_tag(tags=None, args=None, as_str=False):
    if args is None: args = main_args
    if tags is None: tags = args["tag"]
    tag_dict = {}
    tag_str = ""
    for _t in tags:
        if _t in args:
            val = args[_t]
            if type(val) == list: val = "@".join(val)
            val = str(val).split("/")[-1]
            tag_dict[_t] = val
            tag_str += "|" + _t + "=" + val
        else:
            tag_dict[_t] = ""
    if as_str:
        return tag_str
    return tag_dict

tehran = timezone('Asia/Tehran')
now = datetime.datetime.now(tehran)
now = now.strftime('%Y-%m-%d-%H:%M')
home = expanduser("~")
colab = not "ahmad" in home and not "pouramini" in home
if not colab: 
    logPath = os.path.join(home, "logs")
    resPath = os.path.join(home, "results") 
    pretPath = os.path.join(home, "pret") 
else:
    home = "/content/drive/MyDrive/"
    pretPath = "/content/drive/MyDrive/pret"
    logPath = "/content/drive/MyDrive/logs"
    resPath = "/content/drive/MyDrive/logs/results"

pp = Path(__file__).parent.parent.resolve()
dataPath = os.path.join(pp, "data", "atomic2020")
confPath = "base_confs" 

Path(resPath).mkdir(exist_ok=True, parents=True)
Path(logPath).mkdir(exist_ok=True, parents=True)

#logFilename = os.path.join(logPath, "all.log") #app_path + '/log_file.log'
FORMAT = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)10s() ] %(message)s")
FORMAT2 = logging.Formatter("%(message)s")
#logging.basicConfig(filename=logFilename)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(FORMAT2)
mlog = logging.getLogger("att.main")
mlog.setLevel(logging.INFO)
mlog.addHandler(consoleHandler)
clog = logging.getLogger("att.cfg")
dlog = logging.getLogger("att.data")
vlog = logging.getLogger("att.eval")
tlog = logging.getLogger("att.train")
timelog = logging.getLogger("att.time")
plog = logging.getLogger("att.preview")

def getFname(name, path=""):
    if not path:
        if "ahmad" in home or "pouramini" in home:
            path = os.path.join(home, "logs")
        else:
            path = "/content"
    logFilename = os.path.join(path, f"{name}.log")
    return logFilename

def tinfo(text, *args, **kwargs):
    tlog.info(text, *args)


import inspect
import sys
BREAK_POINT = 0

def setbp(bpoint):
    global BREAK_POINT
    BREAK_POINT=bpoint

def bp(break_point):
    global BREAK_POINT
    if colab: return
    equal = False
    if str(BREAK_POINT).startswith("="):
        equal = True
    BREAK_POINT = str(BREAK_POINT).strip("=") 
    cond = BREAK_POINT in str(break_point) 
    if not equal:
        cond = cond or str(break_point) in BREAK_POINT 
    if cond:
        fname = sys._getframe().f_back.f_code.co_name
        line = sys._getframe().f_back.f_lineno
        mlog.info("break point at %s line %s",fname, line)
        breakpoint()

def trace(frame, event, arg):
    if event == "call":
        filename = frame.f_code.co_filename
        if filename.endswith("train/train.py"):
            lineno = frame.f_lineno
            # Here I'm printing the file and line number,
            # but you can examine the frame, locals, etc too.
            print("%s @ %s" % (filename, lineno))
    return trace

mlog.info(now)
#sys.settrace(trace)
def add_handler(logger, fname, set_format=False):
    logger.setLevel(logging.INFO)
    logFilename = os.path.join("logs", fname + ".log")
    handler = logging.FileHandler(logFilename, mode="w")
    if set_format:
        handler.setFormatter(FORMAT)
    logger.addHandler(handler)
    return logFilename

Path("logs").mkdir(parents=True, exist_ok=True)

for logger, fname in zip([mlog,dlog,clog,vlog,tlog,timelog], ["main","data","cfg","eval","train", "time"]):
    add_handler(logger, fname)

def set_args(args):
    global main_args 
    main_args =args
    tlog.handlers.clear()
    tags = "_".join(list(get_tag(args["tag"]).values()))
    exp = str(args["expid"]) + "_" + tags 
    tHandler = logging.FileHandler(getFname(exp + "_time", 
        path=args["save_path"]), mode='w')
    tHandler.setFormatter(FORMAT)
    tlog.addHandler(tHandler)
    tlog.setLevel(logging.INFO)


