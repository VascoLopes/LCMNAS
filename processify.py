import os
import _pickle as cPickle
import time

def processify(func, *args, **kwds):
    pread, pwrite = os.pipe()
    pid = os.fork()
    if pid > 0:
        os.close(pwrite)
        with os.fdopen(pread, 'rb') as f:
            status, result = cPickle.load(f)
        os.waitpid(pid, 0)
        if status == 0:
            return result
        else:
            raise result
    else: 
        os.close(pread)
        try:
            result = func(*args, **kwds)
            status = 0
        except Exception as exc:
            result = exc
            status = 1
        with os.fdopen(pwrite, 'wb') as f:
            print()
            try:
                cPickle.dump((status,result), f, -1)
            except cPickle.PicklingError as exc:
                cPickle.dump((2,exc), f, -1)
            #except Exception as exc:
            #    cPickle.dump((3,exc), f, -1)
        os._exit(0)


'''
import os
import sys
import traceback
from functools import wraps
from multiprocessing import Process, Queue


def processify(func):
    #Decorator to run a function as a process.
    #Be sure that every argument and the return value
    #is *pickable*.
    #The created process is joined, so the code does not
    #run in parallel.
    

    def process_func(q, *args, **kwargs):
        try:
            ret = func(*args, **kwargs)
        except Exception:
            ex_type, ex_value, tb = sys.exc_info()
            error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
            ret = None
        else:
            error = None

        q.put((ret, error))

    # register original function with different name
    # in sys.modules so it is pickable
    process_func.__name__ = func.__name__ + 'processify_func'
    setattr(sys.modules[__name__], process_func.__name__, process_func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        q = Queue()
        p = Process(target=process_func, args=[q] + list(args), kwargs=kwargs)
        p.start()
        ret, error = q.get()

        if error:
            ex_type, ex_value, tb_str = error
            message = '%s (in subprocess)\n%s' % (ex_value.message, tb_str)
            raise ex_type(message)

        return ret
    return wrapper
'''