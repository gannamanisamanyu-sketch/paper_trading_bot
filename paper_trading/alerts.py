"""Optional notification system.

Alerts are informational only. They NEVER place or route orders.
Channels:
* console + ``data/paper/alerts.csv`` (always on when alerts are enabled)
* macOS banner notifications via ``osascript`` (``alerts.macos_notifications``)
* email via SMTP (``alerts.email.enabled``). The password is read from an
  environment variable and never stored in the config file.
"""
from __future__ import annotations

import logging
import os
import smtplib
import subprocess
import sys
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

# alert types
BUY_SIGNAL = "BUY_SIGNAL"
SELL_SIGNAL = "SELL_SIGNAL"
STOP_LOSS = "STOP_LOSS"
DRAWDOWN = "DRAWDOWN"
RISK_LIMIT = "RISK_LIMIT"
REBALANCE = "REBALANCE"


@dataclass
class Alert:
    date: str
    kind: str
    message: str


class AlertManager:
    def __init__(self, acfg, log_path: Path):
        self.cfg = acfg
        self.log_path = log_path
        self.alerts: List[Alert] = []

    def add(self, date, kind: str, message: str) -> None:
        if not self.cfg.enabled:
            return
        self.alerts.append(Alert(str(pd.Timestamp(date).date()), kind, message))

    def dispatch(self) -> List[Alert]:
        if not self.alerts:
            return []
        for a in self.alerts:
            logger.info("ALERT [%s] %s", a.kind, a.message)
        self._append_csv()
        if self.cfg.macos_notifications and sys.platform == "darwin":
            self._macos()
        if self.cfg.email.enabled:
            self._email()
        sent, self.alerts = self.alerts, []
        return sent

    # ------------------------------------------------------------------ channels
    def _append_csv(self) -> None:
        df = pd.DataFrame([a.__dict__ for a in self.alerts])
        header = not self.log_path.exists()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.log_path, mode="a", header=header, index=False)

    def _macos(self) -> None:
        summary = f"{len(self.alerts)} alert(s): " + "; ".join(a.kind for a in self.alerts[:4])
        body = self.alerts[0].message.replace('"', "'")[:180]
        try:
            subprocess.run(["osascript", "-e",
                            f'display notification "{body}" with title "Paper Trading Bot" subtitle "{summary}"'],
                           check=False, timeout=10)
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("macOS notification failed: %s", exc)

    def _email(self) -> None:
        e = self.cfg.email
        password = os.environ.get(e.password_env_var, "")
        if not (e.sender and e.recipient and password):
            logger.warning("Email alerts enabled but sender/recipient/%s not set; skipping",
                           e.password_env_var)
            return
        msg = EmailMessage()
        msg["Subject"] = f"[Paper Trading Bot] {len(self.alerts)} alert(s)"
        msg["From"], msg["To"] = e.sender, e.recipient
        msg.set_content("\n".join(f"{a.date} [{a.kind}] {a.message}" for a in self.alerts)
                        + "\n\nPaper trading only. No orders were sent to any broker.")
        try:
            with smtplib.SMTP(e.smtp_server, e.smtp_port, timeout=20) as s:
                s.starttls()
                s.login(e.sender, password)
                s.send_message(msg)
        except (OSError, smtplib.SMTPException) as exc:
            logger.warning("Email alert failed: %s", exc)
