import json
import socket
import threading
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from urllib.request import urlopen

import uvicorn
from playwright.sync_api import Page, expect, sync_playwright

from adaptive_jump.monitor.api import MonitorServices, create_app
from adaptive_jump.monitor.audit import AuditStore
from adaptive_jump.monitor.event_store import EventStore
from adaptive_jump.monitor.events import ResearchEvent
from adaptive_jump.monitor.evidence import OutcomeLocked
from adaptive_jump.monitor.http_security import HttpSecurityConfig, RequestSecurity
from adaptive_jump.monitor.queue import QueueStore, StudyDefinition
from adaptive_jump.monitor.security import (
    AuthenticationError,
    LocalAuthenticator,
    Principal,
)


class _Authenticator:
    credential_header = "Cf-Access-Jwt-Assertion"
    challenge = "Cloudflare-Access"

    def authenticate(self, assertion):
        if assertion == "owner-token":
            return Principal("owner@example.com", "owner")
        if assertion == "viewer-token":
            return Principal("advisor@example.com", "viewer")
        raise AuthenticationError("invalid fixture assertion")


class _Evidence:
    runs = (
        {"run_id": "open-run", "title": "Verified baseline", "available": True},
        {"run_id": "locked-run", "title": "Boundary-stopped study", "available": True},
    )

    def catalog(self):
        return self.runs

    def evidence(self, run_id):
        opened = run_id == "open-run"
        return {
            "run_id": run_id,
            "title": "Verified baseline" if opened else "Boundary-stopped study",
            "status": "complete" if opened else "boundary_failed",
            "metrics_opened": opened,
            "claim_label": "ENGINEERING FIXTURE",
            "verification": {
                "inventory_files": 12 if opened else 8,
                "boundary_rows": 2,
                "metric_rows": 9 if opened else 0,
                "maximum_metric_absolute_difference": 0.0,
            },
            "boundaries": [
                {
                    "market": "us",
                    "model": "fixed_jm",
                    "delay": 1,
                    "upper_candidate": 1200.0,
                    "selected_months": 2 if opened else 6,
                    "total_months": 100,
                    "passed": opened,
                },
                {
                    "market": "de",
                    "model": "fixed_jm",
                    "delay": 5,
                    "upper_candidate": 1200.0,
                    "selected_months": 1 if opened else 7,
                    "total_months": 100,
                    "passed": opened,
                },
            ],
        }

    def outcome(self, run_id):
        if run_id != "open-run":
            raise OutcomeLocked("fixture outcomes remain locked")
        metrics = []
        for market, offset in (("us", 0.0), ("de", -0.1), ("jp", -0.2)):
            for model, sharpe in (
                ("buy_and_hold", 0.5 + offset),
                ("hmm", 0.6 + offset),
                ("fixed_jm", 0.7 + offset),
            ):
                metrics.append(
                    {"market": market, "model": model, "delay": 1, "sharpe": sharpe}
                )
        return {
            "run_id": run_id,
            "title": "Verified baseline",
            "verification": {"status": "complete"},
            "metrics": metrics,
            "claim": {"conclusion": "ENGINEERING fixture conclusion"},
        }

    def market_data(self, _run_id, _market):
        return json.loads(_market_payload())

    def market_story(self, _run_id, market, model, delay):
        source_rows = json.loads(_market_payload())["rows"][1:]
        signals = [0, 1, 1, 0, 0]
        positions = [0, 0, 0, 1, 1]
        strategy_wealth = [100.01, 100.02, 100.03, 102.99, 101.96]
        buy_hold_wealth = [101.0, 102.0, 103.0, 102.0, 100.0]
        rows = []
        for index, source in enumerate(source_rows):
            rows.append(
                {
                    "date": source["date"],
                    "excess_return": [0.009, 0.009, 0.009, -0.011, -0.021][index],
                    "dd_10": [0.01, 0.009, 0.008, 0.012, 0.018][index],
                    "sortino_20": [0.2, 0.3, 0.4, 0.1, -0.2][index],
                    "sortino_60": [0.1, 0.15, 0.2, 0.12, 0.0][index],
                    "equity_simple": [0.01, 0.0099, 0.0098, -0.0097, -0.0196][index],
                    "signal": signals[index],
                    "position": positions[index],
                    "one_way_turnover": 1.0 if index == 3 else 0.0,
                    "transaction_cost": 0.001 if index == 3 else 0.0,
                    "strategy_return": [0.0001, 0.0001, 0.0001, 0.029, -0.01][index],
                    "strategy_wealth_100": strategy_wealth[index],
                    "strategy_drawdown": strategy_wealth[index]
                    / max(strategy_wealth[: index + 1])
                    - 1,
                    "buy_hold_return": [0.01, 0.0099, 0.0098, -0.0097, -0.0196][index],
                    "buy_hold_wealth_100": buy_hold_wealth[index],
                    "buy_hold_drawdown": buy_hold_wealth[index]
                    / max(buy_hold_wealth[: index + 1])
                    - 1,
                }
            )
        return {
            "run_id": "fixed-baselines-fixture",
            "market": market,
            "model": model,
            "protocol": {
                "delay_trading_days": delay,
                "effective_return_offset": delay + 1,
                "one_way_cost_bps": 10,
            },
            "coverage": {
                "first_date": rows[0]["date"],
                "last_date": rows[-1]["date"],
                "rows": len(rows),
            },
            "rows": rows,
        }


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@contextmanager
def _monitor_origin(tmp_path: Path, authenticator=None):
    port = _free_port()
    origin = f"http://127.0.0.1:{port}"
    runtime = tmp_path / "artifacts/.monitor"
    studies = {"study-a": StudyDefinition("study-a", "replication")}
    queue = QueueStore(runtime / "control.sqlite3", studies)
    active = queue.enqueue("study-a")
    queue.claim_next()
    events = EventStore(runtime)
    observer = events.observer(active.job_id)
    observer(ResearchEvent("process_started", "worker", payload={"pid": 321}))
    observer(
        ResearchEvent(
            "stage_started", "hmm", completed=0, total=5, payload={"fit_window": 3000}
        )
    )
    observer(
        ResearchEvent(
            "terminal_state",
            "hmm",
            visibility="decision",
            market="us",
            model="hmm",
            date=date(2023, 12, 29),
            completed=1,
            total=5,
            payload={
                "state": 1,
                "features": {
                    "dd_10": 0.12,
                    "sortino_20": -0.4,
                    "sortino_60": 0.3,
                    "excess_return": -0.01,
                },
            },
        )
    )
    observer(
        ResearchEvent(
            "resource_sample",
            "worker",
            payload={"cpu_percent": 32.0, "rss_bytes": 268435456, "process_count": 2},
        )
    )
    observer(
        ResearchEvent(
            "selection_checkpoint",
            "selection",
            visibility="decision",
            market="us",
            model="fixed_jm",
            delay=1,
            date=date(2023, 12, 29),
            payload={
                "completed_months": 10,
                "selected_candidate": 35.0,
                "cv_surface": [
                    {
                        "candidate": 5.0,
                        "valid_returns": 2000,
                        "sharpe": 0.4,
                        "eligible": True,
                    },
                    {
                        "candidate": 35.0,
                        "valid_returns": 2000,
                        "sharpe": 0.7,
                        "eligible": True,
                    },
                ],
            },
        )
    )
    observer(
        ResearchEvent(
            "selected_signal",
            "selection",
            visibility="decision",
            market="us",
            model="fixed_jm",
            delay=1,
            date=date(2023, 12, 29),
            payload={
                "decision_date": "2023-12-29",
                "selected_candidate": 35.0,
                "signal": 1,
                "scheduled_position": 1,
                "effective_return_offset": 2,
            },
        )
    )
    observer(
        ResearchEvent(
            "boundary_diagnostic",
            "selection",
            visibility="decision",
            market="us",
            model="fixed_jm",
            delay=1,
            payload={
                "upper_candidate": 1200.0,
                "selected_months": 6,
                "total_months": 100,
                "fraction": 0.06,
                "limit": 0.05,
                "passed": False,
            },
        )
    )
    observer(
        ResearchEvent(
            "artifact_verified",
            "verification",
            visibility="decision",
            payload={"run_id": "fixed-baselines-fixture", "status": "complete"},
        )
    )
    queue.finish(active.job_id, "succeeded", 0)
    services = MonitorServices(
        queue=queue,
        events=events,
        evidence=_Evidence(),
        audit=AuditStore(runtime),
        authenticator=authenticator or _Authenticator(),
        request_security=RequestSecurity(
            HttpSecurityConfig(origin, b"x" * 32), nonce_factory=lambda: "fixture"
        ),
    )
    app = create_app(services)

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{origin}/healthz", timeout=0.2) as response:
                if response.status == 200:
                    break
        except OSError:
            time.sleep(0.05)
    else:
        raise RuntimeError("fixture monitor did not start")
    try:
        yield origin, active.job_id
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        assert not thread.is_alive()


def _block_external(context, origin: str) -> None:
    context.route(
        "**/*",
        lambda route: (
            route.continue_()
            if route.request.url.startswith(origin)
            or route.request.url.startswith("data:")
            else route.abort()
        ),
    )


def _watch_errors(page: Page) -> list[str]:
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on(
        "console",
        lambda message: (
            errors.append(message.text) if message.type == "error" else None
        ),
    )
    return errors


def _market_payload() -> str:
    prices = (
        ("2023-12-22", 99, 101, 98, 100),
        ("2023-12-25", 100, 102, 99, 101),
        ("2023-12-26", 101, 103, 100, 102),
        ("2023-12-27", 102, 104, 101, 103),
        ("2023-12-28", 103, 104, 101, 102),
        ("2023-12-29", 102, 103, 99, 100),
    )
    rows = [
        {
            "date": market_date,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 0,
        }
        for market_date, open_, high, low, close in prices
    ]
    return json.dumps(
        {
            "run_id": "fixed-baselines-fixture",
            "market": "us",
            "source": {"provider": "yahoo", "source_id": "^SP500TR", "deviations": []},
            "coverage": {
                "first_date": rows[0]["date"],
                "last_date": rows[-1]["date"],
                "rows": len(rows),
            },
            "quality": {
                "candles_available": True,
                "volume_available": False,
                "distinct_ohlc_rows": 6,
            },
            "rows": rows,
        }
    )


def _assert_no_horizontal_overflow(page: Page) -> None:
    overflow = page.evaluate(
        "() => ({page: document.documentElement.scrollWidth - window.innerWidth, "
        "elements: [...document.querySelectorAll('body *')]"
        ".filter(node => node.getBoundingClientRect().right > window.innerWidth + 1)"
        ".slice(0, 8).map(node => `${node.tagName}.${node.className}`)})"
    )
    assert overflow["page"] <= 0, overflow


def _assert_canvas_has_pixels(page: Page, selector: str) -> None:
    canvases = page.locator(selector)
    expect(canvases.first).to_be_visible()
    assert canvases.evaluate_all(
        "items => items.some(c => c.width > 10 && c.height > 10 && "
        "Array.from(c.getContext('2d').getImageData(0,0,c.width,c.height).data)"
        ".some((value,index) => index % 4 === 3 && value > 0))"
    )


def test_monitor_ui_in_real_chromium_desktop_mobile_and_no_js(tmp_path: Path) -> None:
    with _monitor_origin(tmp_path) as (origin, active_job_id), sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        owner = browser.new_context(
            viewport={"width": 1440, "height": 900},
            extra_http_headers={"Cf-Access-Jwt-Assertion": "owner-token"},
        )
        _block_external(owner, origin)
        page = owner.new_page()
        errors = _watch_errors(page)
        page.goto(origin, wait_until="domcontentloaded")
        expect(page.locator("#identity")).to_contain_text("owner@example.com · owner")
        expect(page.locator("#summary-study")).to_have_text("study-a")
        expect(page.locator("#cv-body")).to_contain_text("Selected")
        expect(page.locator("#state-body")).to_contain_text("t+2")
        expect(page.locator("#live-boundary-body")).to_contain_text("Expand grid")
        _assert_canvas_has_pixels(page, "#resource-chart canvas")
        page.screenshot(path=tmp_path / "monitor-live.png", full_page=True)

        page.get_by_role("button", name="Queue").click()
        page.get_by_role("button", name="Enqueue").click()
        expect(page.locator("#queue-count")).to_have_text("1 queued")
        _assert_no_horizontal_overflow(page)

        page.get_by_role("button", name="Replay").click()
        page.locator("#replay-job").select_option(active_job_id)
        expect(page.locator("#replay-position")).to_have_text("2023-12-25 · 1 / 5")
        expect(page.locator("#audit-position")).to_have_text("1 / 8")
        expect(page.locator("#replay-summary-close")).to_have_text("101")
        expect(page.locator("#replay-summary-return")).to_have_text("1%")
        expect(page.locator("#decision-input")).to_contain_text("DD-10 1%")
        expect(page.locator("#decision-signal")).to_have_text("Cash · 0")
        expect(page.locator("#decision-delay")).to_have_text("Applies at t+2")
        expect(page.locator("#decision-outcome")).to_contain_text("$100.01")
        expect(page.locator("#replay-frame-row")).to_contain_text("100.01")
        _assert_canvas_has_pixels(page, "#replay-market-chart canvas")
        _assert_canvas_has_pixels(page, "#replay-feature-chart canvas")
        initial_markers = page.evaluate(
            "() => echarts.getInstanceByDom("
            "document.getElementById('replay-market-chart')).getOption().series"
            ".filter(series => ['Enter market', 'Move to cash'].includes(series.name))"
            ".flatMap(series => series.data)"
        )
        assert initial_markers == []
        page.wait_for_timeout(1000)
        expect(page.locator("#replay-position")).to_have_text("2023-12-25 · 1 / 5")
        page.locator("#replay-model").select_option("hmm")
        expect(page.locator("#decision-input")).to_contain_text("Total return 1%")
        page.locator("#replay-delay").select_option("5")
        expect(page.locator("#decision-delay")).to_have_text("Applies at t+6")
        page.locator("#replay-speed").select_option("0.5")
        page.locator('[data-replay="play"]').click()
        expect(page.locator("#replay-position")).to_have_text(
            "2023-12-26 · 2 / 5", timeout=3000
        )
        page.locator('[data-replay="pause"]').click()
        page.locator("#replay-speed").select_option("5")
        page.locator('[data-replay="play"]').click()
        expect(page.locator("#replay-position")).to_have_text(
            "2023-12-29 · 5 / 5", timeout=8000
        )
        trade_markers = page.evaluate(
            "() => echarts.getInstanceByDom("
            "document.getElementById('replay-market-chart'))"
            ".getOption().series.find(series => series.name === 'Enter market').data"
        )
        assert len(trade_markers) == 1
        page.locator('[data-replay="reset"]').click()
        expect(page.locator("#replay-position")).to_have_text("2023-12-25 · 1 / 5")

        page.get_by_role("button", name="Compare").click()
        expect(page.locator("#compare-status")).to_have_text("2 / 2 verified")
        _assert_canvas_has_pixels(page, "#comparison-chart canvas")
        page.get_by_role("button", name="Evidence").click()
        page.get_by_role("button", name="Boundary-stopped study").click()
        expect(page.locator("#evidence-lock")).to_contain_text("locked")
        expect(page.locator("#boundary-body")).to_contain_text("Expand grid")
        _assert_no_horizontal_overflow(page)
        page.screenshot(path=tmp_path / "monitor-desktop.png", full_page=True)
        assert errors == []
        owner.close()

        viewer = browser.new_context(
            viewport={"width": 390, "height": 844},
            extra_http_headers={"Cf-Access-Jwt-Assertion": "viewer-token"},
        )
        _block_external(viewer, origin)
        mobile = viewer.new_page()
        mobile_errors = _watch_errors(mobile)
        mobile.goto(origin, wait_until="domcontentloaded")
        expect(mobile.locator("#identity")).to_contain_text(
            "advisor@example.com · viewer"
        )
        mobile.get_by_role("button", name="Queue").click()
        expect(mobile.get_by_role("button", name="Enqueue")).to_be_disabled()
        expect(mobile.get_by_role("button", name="Cancel")).to_have_count(0)
        for view in ("Live", "Replay", "Compare", "Evidence"):
            mobile.get_by_role("button", name=view, exact=True).click()
            _assert_no_horizontal_overflow(mobile)
        mobile.screenshot(path=tmp_path / "monitor-mobile.png", full_page=True)
        assert mobile_errors == []
        viewer.close()

        no_js = browser.new_context(
            viewport={"width": 390, "height": 844},
            java_script_enabled=False,
            extra_http_headers={"Cf-Access-Jwt-Assertion": "owner-token"},
        )
        _block_external(no_js, origin)
        fallback = no_js.new_page()
        fallback.goto(origin, wait_until="domcontentloaded")
        expect(fallback.locator(".noscript")).to_contain_text(
            "JavaScript is required for authenticated live data."
        )
        _assert_no_horizontal_overflow(fallback)
        no_js.close()
        browser.close()


def test_boundary_failed_job_keeps_audit_without_requesting_market_replay(
    tmp_path: Path,
) -> None:
    with _monitor_origin(tmp_path) as (origin, active_job_id), sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            extra_http_headers={"Cf-Access-Jwt-Assertion": "owner-token"}
        )
        _block_external(context, origin)
        page = context.new_page()
        errors = _watch_errors(page)
        market_requests = []

        def rewrite_events(route):
            response = route.fetch()
            payload = response.json()
            for event in payload["events"]:
                if event["kind"] == "artifact_verified":
                    event["payload"] = {
                        "run_id": "grid-eval-fixture",
                        "status": "boundary_failed",
                    }
            route.fulfill(
                status=response.status,
                content_type="application/json",
                body=json.dumps(payload),
            )

        page.route(f"{origin}/api/jobs/{active_job_id}/events", rewrite_events)
        page.on(
            "request",
            lambda request: (
                market_requests.append(request.url)
                if "/markets/" in request.url
                else None
            ),
        )
        page.goto(origin, wait_until="domcontentloaded")
        page.get_by_role("button", name="Replay").click()

        message = (
            "Runtime audit is available, but market replay requires a completed "
            "fixed-baseline artifact."
        )
        expect(page.locator("#replay-source")).to_have_text(message)
        expect(page.locator("#audit-count")).to_have_text("8 append-only events")
        expect(page.locator("#audit-position")).to_have_text("1 / 8")
        assert market_requests == []
        assert errors == []

        context.close()
        browser.close()


def test_local_owner_opens_the_monitor_with_browser_basic_auth(tmp_path: Path) -> None:
    password = "correct-local-password"
    authenticator = LocalAuthenticator(password)
    with (
        _monitor_origin(tmp_path, authenticator) as (origin, active_job_id),
        sync_playwright() as pw,
    ):
        browser = pw.chromium.launch(headless=True)

        anonymous = browser.new_page()
        response = anonymous.goto(origin)
        assert response is not None and response.status == 401
        anonymous.close()

        context = browser.new_context(
            http_credentials={"username": "owner", "password": password}
        )
        _block_external(context, origin)
        page = context.new_page()
        event_requests = []
        page.on(
            "request",
            lambda request: (
                event_requests.append(request.url)
                if request.url.endswith(f"/api/jobs/{active_job_id}/events")
                else None
            ),
        )
        errors = _watch_errors(page)
        response = page.goto(origin, wait_until="domcontentloaded")
        assert response is not None and response.status == 200
        expect(page.locator("#identity")).to_have_text("local-owner@localhost · owner")
        page.get_by_role("button", name="Replay").click()
        expect(page.locator("#audit-position")).to_have_text("1 / 8")
        assert len(event_requests) == 1
        page.get_by_role("button", name="Queue").click()
        expect(page.get_by_role("button", name="Enqueue")).to_be_enabled()
        _assert_no_horizontal_overflow(page)
        assert errors == []
        context.close()
        browser.close()


def test_monitor_ui_names_an_empty_study_catalog(tmp_path: Path) -> None:
    with _monitor_origin(tmp_path) as (origin, _), sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            extra_http_headers={"Cf-Access-Jwt-Assertion": "owner-token"}
        )
        _block_external(context, origin)
        page = context.new_page()
        page.route(
            f"{origin}/api/studies",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body='{"queueable":[]}',
            ),
        )
        page.goto(origin, wait_until="domcontentloaded")
        page.get_by_role("button", name="Queue").click()
        expect(page.locator("#study-select")).to_be_disabled()
        expect(page.locator("#study-select")).to_have_text(
            "No FROZEN studies available"
        )
        expect(page.get_by_role("button", name="Enqueue")).to_be_disabled()
        context.close()
        browser.close()
