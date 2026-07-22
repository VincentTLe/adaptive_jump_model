# Monitor Deployment

This optional procedure deploys the monitor for authenticated remote access.
Normal local use needs no Cloudflare account and is documented in the README.
In Cloudflare mode the Python origin listens only on `127.0.0.1:8765`; the
tunnel is the only remote ingress. Runtime files stay under ignored
`artifacts/.monitor/`.

## 1. Install The Locked Python Environment

```bash
uv sync --locked --extra data --extra monitor
uv run python -c "import fastapi, jwt, psutil, uvicorn"
uv lock --check
uv pip check --python .venv/bin/python
```

The optional `monitor` group is pinned in `pyproject.toml` and `uv.lock`. Do not
create a second requirements file.

## 2. Install The Pinned Tunnel Binary

The commands below are for this host's `x86_64` Linux architecture. They pin the
official `2026.6.0` release and verify the release-page checksum before install.

```bash
CLOUDFLARED_VERSION=2026.6.0
CLOUDFLARED_SHA256=08d27c4c5d3ed73ee3e98ef2ddceb4ad09fd4cfc28e243565a189538e8ccd706
curl --fail --location --output /tmp/cloudflared-linux-amd64 \
  "https://github.com/cloudflare/cloudflared/releases/download/${CLOUDFLARED_VERSION}/cloudflared-linux-amd64"
echo "${CLOUDFLARED_SHA256}  /tmp/cloudflared-linux-amd64" | sha256sum --check --strict
sudo install -o root -g root -m 0755 /tmp/cloudflared-linux-amd64 /usr/local/bin/cloudflared
/usr/local/bin/cloudflared --version
```

Release identity and checksums come from the
[official 2026.6.0 release](https://github.com/cloudflare/cloudflared/releases/tag/2026.6.0).
Never replace the version with `latest` in an installation command.

## 3. Create The Named Tunnel

```bash
cloudflared tunnel login
cloudflared tunnel create adaptive-jump-monitor
cloudflared tunnel route dns adaptive-jump-monitor monitor.example.com
install -m 0600 deploy/cloudflared-config.yml.example /home/tle/.cloudflared/config.yml
```

Replace `TUNNEL_UUID`, credentials path, and hostname in the copied file. Keep
the final catch-all `http_status:404` ingress rule. Cloudflare's
[Linux service guide](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/local-management/as-a-service/linux/)
requires the tunnel UUID and credentials file for a locally managed service.

```bash
sudo cloudflared --config /home/tle/.cloudflared/config.yml service install
sudo systemctl enable --now cloudflared
```

## 4. Protect The Hostname With Access

In Cloudflare One, create a self-hosted application for the complete monitor
hostname. Create one `Allow` policy whose `Include` selector lists only the
owner and advisor email addresses. Require the One-time PIN login method. Do
not use `Everyone`, an email-domain wildcard, or `Login Methods: One-time PIN`
as the Include rule.

Cloudflare documents that Access is deny-by-default and that broad OTP Include
rules can admit every valid email. See
[Access policies](https://developers.cloudflare.com/cloudflare-one/access-controls/policies/)
and [One-time PIN login](https://developers.cloudflare.com/cloudflare-one/integrations/identity-providers/one-time-pin/).

Record the application AUD tag and team-domain issuer. The origin independently
validates the assertion header as recommended by Cloudflare's
[JWT validation guide](https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/authorization-cookie/validating-json/).

## 5. Install Origin Secrets

```bash
install -d -m 0700 /home/tle/.config/adaptive-jump
install -m 0600 deploy/monitor.env.example /home/tle/.config/adaptive-jump/monitor.env
openssl rand -base64 48 | tr '+/' '-_' | tr -d '=\n'
```

Replace every placeholder in the copied environment file. Keep
`ADAPTIVE_JUMP_MONITOR_ACCESS=cloudflare`; without this explicit setting the
monitor intentionally starts in local mode. The public origin
must exactly match the HTTPS hostname, with no path or trailing slash. The owner
email and each comma-separated viewer email must exactly match Access claims.
Use the final command's output as `ADAPTIVE_JUMP_CSRF_SECRET`.

## 6. Start The Monitor

```bash
sudo install -o root -g root -m 0644 \
  deploy/adaptive-jump-monitor.service \
  /etc/systemd/system/adaptive-jump-monitor.service
sudo systemctl daemon-reload
sudo systemctl enable --now adaptive-jump-monitor
curl --fail http://127.0.0.1:8765/healthz
sudo systemctl status adaptive-jump-monitor cloudflared
```

The unit uses `KillMode=control-group` with a 55-second stop window. The
supervisor first sends `SIGINT`, preserving the latest completed atomic
checkpoint, then escalates only if the process does not stop. The queue records
that run as `interrupted`; the owner must explicitly resume it. Work since the
last checkpoint may be repeated. Systemd kills the whole control group only if
the graceful lifecycle itself exceeds the deadline.

## 7. Acceptance And Operations

The local lifecycle acceptance completed on 2026-07-14 using the production
loopback server, an ephemeral HTTPS JWKS issuer, and real Chromium. Cancel,
checkpoint reuse, server shutdown, resume, replay, verifier ordering, and
artifact parity passed. This did not exercise a real Cloudflare account,
hostname, tunnel, Access policy, OTP flow, or owner/viewer deployment; every
item below must still pass on the deployed HTTPS hostname.

Open the HTTPS hostname and authenticate with an exact approved email. Confirm:

1. The viewer can read Live, Replay, Compare, and Evidence but cannot mutate.
2. The owner can enqueue only code-registered studies whose registry status is
   `FROZEN`.
3. A locked run returns `423` for outcomes and remains labeled locked in the UI.
4. `ss -ltnp` shows port `8765` only on `127.0.0.1`.
5. Restarting the monitor preserves queue and event history and terminates any
   matching leftover process before an interrupted run can be resumed.

For VS Code, run **Simple Browser: Show** from the Command Palette and enter the
same HTTPS hostname. Do not forward the Cloudflare-mode origin publicly; it
expects the tunnel to supply a signed Access assertion.

Operational commands:

```bash
journalctl -u adaptive-jump-monitor -f
journalctl -u cloudflared -f
systemctl restart adaptive-jump-monitor
systemctl restart cloudflared
```

There is no delete endpoint. Queue state, mutation audit, and event journals are
retained under `artifacts/.monitor/`; they are operational records, not claim
evidence.
