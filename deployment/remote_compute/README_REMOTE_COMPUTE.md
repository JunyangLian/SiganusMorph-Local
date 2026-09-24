# SiganusMorph V1 remote compute deployment

The Alibaba Cloud host remains the Streamlit frontend and task store. A
networked bioinformatics server pulls jobs over HTTPS, performs calibration and
measurement with one 8-thread worker, and uploads a result package. The worker
does not accept inbound connections.

## Safety boundaries

- Remote jobs are stored only under `results/remote_measurement_jobs/`.
- The worker never writes corrected keypoints or historical batch outputs.
- Remote mode is opt-in through `SIGANUSMORPH_REMOTE_COMPUTE=1`.
- Worker authentication requires a shared token of at least 32 characters.
- Use HTTPS only. Never send the worker token over plain HTTP.

## Cloud host

Install the lightweight API dependencies in the existing V1 environment:

```bash
cd /home/admin/yangzz/SiganusMorph
.venv/bin/python -m pip install -r deployment/remote_compute/requirements-remote-api.txt
openssl rand -hex 32
```

Copy `remote_api.env.example` to `remote_api.env`, insert the generated token,
and set mode `600`. Add both `SIGANUSMORPH_REMOTE_COMPUTE=1` and
`SIGANUS_REMOTE_JOB_ROOT=...` to the Streamlit service environment as well.

Install and start the API service:

```bash
sudo cp deployment/remote_compute/siganusmorph-remote-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now siganusmorph-remote-api
curl -fsS http://127.0.0.1:8766/health
```

Add `baota_nginx_location.conf` inside the HTTPS `server` block managed by
Baota, test the Baota Nginx configuration, and reload that Nginx instance.

## Bioinformatics worker

Copy `worker_runtime.env.example` to
`deployment/remote_worker/worker_runtime.env`, insert the same token, and set
mode `600`. Then run a connection test:

```bash
cd /home/user/Lianjunyang/SiganusMorph
cp deployment/remote_compute/run_remote_worker.sh deployment/remote_worker/
chmod 700 deployment/remote_worker/run_remote_worker.sh
deployment/remote_worker/run_remote_worker.sh --once
```

With no queued job, the command exits normally. For an interactive measurement
period, keep one worker running:

```bash
nohup deployment/remote_worker/run_remote_worker.sh \
  > worker_runtime/logs/worker.log 2>&1 &
echo $! > worker_runtime/worker.pid
```

Stop only this user's worker with:

```bash
kill "$(cat worker_runtime/worker.pid)"
```

Do not start multiple workers on the shared server. Subsequent tasks reuse the
loaded models and remain queued while one image is being measured.
