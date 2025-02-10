from pathlib import Path
import pickle

__all__ = ["PickleFileLoader", "PickleFileWriter", "PickleFileWriterAsync"]

class PickleFileLoader:
    def __init__(self, file_path:Path):
        self.file_path = Path(file_path)
        
    def __iter__(self):
        with self.file_path.open(mode="rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break


class PickleFileWriter:
    def __init__(self, file_path:Path):
        self.file_path = Path(file_path)
        self.file_obj = None
        
    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, obj):
        pickle.dump(obj, self.file_obj)
        
    def close(self):
        self.file_obj.close()

    def open(self):
        self.file_obj = self.file_path.open(mode="wb")
        return self



class PickleFileWriterAsync(PickleFileWriter):
    def __init__(self, file_path:Path, num_writer_worker=1):
        print("Warning!! This Class is not stable. Sometimes objects to be written is desappear")
        super().__init__(file_path)
        self.queue = multiprocessing.JoinableQueue()
        assert num_writer_worker > 0, "Number of worker must be more than 1 or equal."
        self.num_writer_worker = num_writer_worker
        self.workers = []
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, obj):
        self.queue.put(obj)
        
    def open(self):
        self.file_obj = self.file_path.open(mode="wb")
        for _ in range(self.num_writer_worker):
            worker = multiprocessing.Process(
                target=self.write_process,
            )
            worker.start()
            self.workers.append(worker)
        return self

    def close(self):
        self.queue.join()
        self.file_obj.close()
        for worker in self.workers:
            worker.terminate()
            
    def write_process(self):
        while True:
            obj = self.queue.get()
            pickle.dump(obj, self.file_obj)
            self.queue.task_done()

        
