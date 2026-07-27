# Process Instance Lock

## Definition

Global process-level lock ensuring at most one ds4 process runs system-wide. Acquired early in `ds4_engine_open_internal` before any model load, released in `ds4_engine_close` as final cleanup. Uses `flock(fd, LOCK_EX | LOCK_NB)` on a well-known path — hard fail with `exit(2)` if another instance holds the lock.

## Why It Exists

Model maps tens of GiB (GGUF mmap, GPU allocations). Accidental second ds4 process doubles memory, corrupts shared state (GPU page tables, mmap'd files), produces hard-to-debug failures. Lock catches this at startup with clear error message and PID of conflicting process.

## Where It Appears

| File | Symbol | Role |
|---|---|---|
| `ds4.c` | `g_ds4_lock_fd` | Global FD for open lock file, initialized to -1 |
| `ds4.c` | `ds4_acquire_instance_lock` | Acquire lock: open, flock, write PID, register atexit safety net |
| `ds4.c` | `ds4_release_instance_lock` | Release lock: close FD, set g_ds4_lock_fd = -1 |
| `ds4.c` | `ds4_engine_open_internal` | Calls `ds4_acquire_instance_lock()` first thing (before model load) |
| `ds4.c` | `ds4_engine_close` | Calls `ds4_release_instance_lock()` as last cleanup step |

## Lifecycle

```
ds4_engine_open_internal():
  ...
  ds4_acquire_instance_lock()           // acquire global lock
  // model load, GPU init, etc.
  // ...

ds4_engine_close():
  // GPU cleanup, SSD memory lock release
  ds4_release_instance_lock()           // release global lock
  free(e->directional_steering_dirs)
  free(e->directional_steering_file)
  free(e)
```

Lock is **first** thing acquired in open (before model load, GPU init, config validation). Lock is released in close after GPU/resources cleanup but before freeing the engine struct — it is **not** the last operation in close.

## Lock Semantics

| Scenario | Behavior |
|---|---|
| First process starts | Lock acquired, PID written, proceeds normally |
| Second process starts | `flock(fd, LOCK_EX \| LOCK_NB)` returns EWOULDBLOCK, prints error with PID from lock file, `exit(2)` |
| First process exits normally | `ds4_release_instance_lock` closes FD → flock released, lock file persists (stale content) |
| First process crashes | Kernel closes all FDs on process death → flock released, lock file persists with stale PID |
| First process `exec()` | `FD_CLOEXEC` ensures FD closed → flock released |
| `DS4_LOCK_FILE` overridden | Two ds4 processes can coexist if each uses a different lock path (not recommended) |

## Environment Variables

| Variable | Effect |
|---|---|
| `DS4_LOCK_FILE` | Override lock file path. Unset or empty → default `/tmp/ds4.lock`. |

## Code Pattern

```c
static void ds4_acquire_instance_lock(void) {
    const char *path = getenv("DS4_LOCK_FILE");
    if (!path || !path[0]) path = "/tmp/ds4.lock";

    const int fd = open(path, O_RDWR | O_CREAT, 0600);
    if (fd < 0) {
        fprintf(stderr, "ds4: failed to open lock file %s: %s\n", path, strerror(errno));
        exit(2);
    }
    (void)fcntl(fd, F_SETFD, FD_CLOEXEC);

    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno == EWOULDBLOCK) {
            char buf[64];
            ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
            long owner = -1;
            if (n > 0) {
                buf[n] = '\0';
                char *end = NULL;
                owner = strtol(buf, &end, 10);
            }
            if (owner > 0) {
                fprintf(stderr, "ds4: another ds4 process is already running (pid %ld); refusing to start\n", owner);
            } else {
                fprintf(stderr, "ds4: another ds4 process is already running; refusing to start\n");
            }
            close(fd);
            exit(2);
        }
        fprintf(stderr, "ds4: failed to lock %s: %s\n", path, strerror(errno));
        close(fd);
        exit(2);
    }

    if (ftruncate(fd, 0) != 0) {
        fprintf(stderr, "ds4: failed to truncate lock file %s: %s\n", path, strerror(errno));
        close(fd);
        exit(2);
    }
    dprintf(fd, "%ld\n", (long)getpid());
    g_ds4_lock_fd = fd;
    atexit(ds4_release_instance_lock);
}
```

```c
static void ds4_release_instance_lock(void) {
    if (g_ds4_lock_fd >= 0) {
        close(g_ds4_lock_fd);
        g_ds4_lock_fd = -1;
    }
}
```

## Relationship to Other Concepts

- **Called by**: `ds4_engine_open_internal` (first action), `ds4_engine_close` (last action).
- **Protects**: all resources — GGUF mmap, GPU allocations, shared workspaces, temporary files.
- **Enforced for**: all callers — CLI, server, agent, tests. No skip path.
- **Alternatives**: per-model lock (less safe, doesn't prevent cross-model GPU corruption), named semaphore (less portable, cleanup on crash is OS-dependent).

## Notes

- Lock is **always** acquired regardless of CPU/GPU backend, Metal/CUDA/ROCm, or any other configuration. No opt-out.
- `flock` is per-FD, not per-process. Single process opening lock multiple times on same FD succeeds. `atexit` safety net covers `exit()` and `return from main()`, but not `_exit()` or `SIGKILL` (kernel closes FDs either way).
- Lock file is **never unlinked**. Stale file with old PID overwritten by `ftruncate` + `dprintf` on next acquire. `flock` ownership is kernel-internal — inode remains alive as long as any process holds an FD to it.
- `FD_CLOEXEC` ensures lock released if process calls `exec()`, preventing child from inheriting lock.
- Lock is **advisory** (`flock`). Rogue process can ignore it by bypassing `ds4_engine_open_internal`, but any process using proper API is blocked.

[← Back to Index](../README.md)
