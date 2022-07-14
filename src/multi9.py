from concurrent.futures import ThreadPoolExecutor
import time

def make_udon(kind):
    print('  %sうどんを作ります。' % kind)
    time.sleep(3)
    return kind + 'うどん'

kinds = ['たぬき', 'かけ', 'ざる', 'きつね', '天ぷら', '肉']
executor = ThreadPoolExecutor(max_workers=3)
futures = []

for kind in kinds:
    print('%sうどん オーダー入りました。' % kind)
    future = executor.submit(make_udon, kind)
    futures.append(future)

for future in futures:
    print('%sお待たせしました。' % future.result())

executor.shutdown()