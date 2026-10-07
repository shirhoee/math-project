# 04 Performance Rules

1. Depth model loaded once, never per request.
2. Results cached to disk by image hash plus settings.
3. Heavy work never runs on import.
4. The UI must show a loading state within 200 ms of any action.
5. Worker count limited to `max(1, cpu_count()//2)`.
6. Set OMP/MKL thread counts explicitly.
7. Windows-safe multiprocessing (guard with `if __name__ == "__main__"`, avoid spawning processes that import torch).
