import multiprocessing as mp

def foo(q):
    q.put('hello')

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=foo, args=(q,))
    r = ctx.Process(target=foo, args=(q,))
    p.start()
    r.start()
    print(q.get())
    print(q.get())
    p.join()
    r.join()
    print(q.get())
