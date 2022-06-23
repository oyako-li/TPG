import time
from multiprocessing import Pool

# 並列処理させる関数
def nijou(x):
    print('input: %d' % x)
    time.sleep(2)
    retValue = x * x
    print('double: %d' % (retValue))
    return(retValue)

from multiprocessing import Process
import os

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    info('function f')
    print('hello', name)


if __name__ == "__main__":
    info('main line')
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()
    # p = Pool(10) # プロセス数を4に設定
    # result = p.map(nijou, range(10))  # nijou()に0,1,..,9を与えて並列演算
    # print(result)
    print(os.sched_getaffinity(0))