import tracemalloc
import time
from datetime import datetime
from threading import Thread

# Copyed from https://kuttsun.blogspot.com/2022/02/python-tracemalloc.html
class MemoryTracer:

    def __init__(self, interval=5, top_n=10):
        self.__is_running = False
        self.__snapshot = None
        self.interval = interval
        self.top_n = top_n

    def start(self):
        print(f'{self.__class__.__name__} start')
        self.__is_running = True
        self.__thread = Thread(target=self.__run)
        self.__thread.start()

    def stop(self):
        print(f'{self.__class__.__name__} stopping')
        self.__is_running = False
        self.__thread.join()
        print(f'{self.__class__.__name__} stopped')

    def __run(self):
        while self.__is_running:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('traceback')
            print(f'[ Top {self.top_n} : {datetime.now()}]')
            # for stat in top_stats[:self.top_n]:
            #     print(stat)
            if self.__snapshot != None:
                top_stats = snapshot.compare_to(self.__snapshot, 'traceback')
                print(f'[ Top {self.top_n} Difference ]')
                count = 0
                for stat in top_stats:
                    if "tracemalloc" in str(stat):
                        continue
                    print(stat)
                    count += 1
                    if count >= self.top_n:
                        break
                    
            self.__snapshot = snapshot
            time.sleep(self.interval)

