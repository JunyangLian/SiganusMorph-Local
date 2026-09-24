# SiganusMorph V1 low-resource server mode

This mode is intended for a single-user 2-core/2-GB server. It does not change
the model weights, selector rules, corrected keypoints, or measurement formulas.

## Behavior

- Native numerical libraries use one worker thread.
- Heavy automatic-measurement inference is serialized across browser sessions.
- YOLO, heatmap v0.1, and heatmap v0.5 model caches are released between stages.
- Nested inference debug payloads are not retained in the formal-page session.
- The default local/runtime behavior remains unchanged unless the environment
  variable below is enabled.

## systemd environment

Run `sudo systemctl edit siganusmorph-v1` and use:

```ini
[Service]
Environment="SIGANUSMORPH_LOW_RESOURCE_MODE=1"
Environment="SIGANUSMORPH_RUNTIME_THREADS=1"
Environment="OMP_NUM_THREADS=1"
Environment="MKL_NUM_THREADS=1"
Environment="OPENBLAS_NUM_THREADS=1"
Environment="NUMEXPR_NUM_THREADS=1"
Environment="OPENCV_FOR_THREADS_NUM=1"
Environment="MALLOC_ARENA_MAX=2"
CPUQuota=150%
Nice=10
```

Then apply it:

```bash
sudo systemctl daemon-reload
sudo systemctl restart siganusmorph-v1
curl -fsS http://127.0.0.1:8501/SiganusMorph/_stcore/health
```

## Operational expectations

The first automatic measurement remains CPU-intensive and may take several
minutes. CPU utilization near 150% is expected because of the service quota.
Additional users wait for the active inference instead of loading duplicate
models. Do not repeatedly click the measurement button while a job is active.

Monitor one run with:

```bash
watch -n 2 'free -h; echo; ps -C streamlit -o pid,%cpu,%mem,rss,etime,cmd --sort=-rss'
```

If the service stops responding, collect diagnostics before restarting:

```bash
sudo journalctl -u siganusmorph-v1 -n 150 --no-pager
sudo dmesg -T | grep -Ei 'oom|killed process|out of memory' | tail -30
```
