from multiprocessing import Process, Value, Array
import time

def f(n, a):
    n.value = 3.1415927
    for i in range(len(a)):
        time.sleep(1)
        a[i] = -a[i]
def g(n, a):
    while True:
        time.sleep(0.5)
        print(n.value)
        print(a[:])

if __name__ == '__main__':
    num = Value('d', 0.0)
    arr = Array('i', range(10))

    p = Process(target=f, args=(num, arr))
    r = Process(target=g, args=(num, arr))
    p.start()
    r.start()
    p.join(timeout=5)
    r.join(timeout=5)
    p.terminate()
    r.terminate()

    # print(num.value)
    # print(arr[:])