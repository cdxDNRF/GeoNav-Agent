"""Exclusive ownership of a persisted experiment store, independent of GUI."""
import os
from pathlib import Path

class StoreLock:
    """Prevent a second service from interpreting another process's live job as interrupted."""
    def __init__(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        self.handle=(path/'service.lock').open('a+b')
        self.handle.seek(0);self.handle.write(b'0');self.handle.flush();self.handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.handle.close();raise ValueError('该工作台目录已有服务使用；不要同时启动第二份') from None
    def close(self):self.handle.close()


