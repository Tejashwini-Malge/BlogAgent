"""
Gmail SMTP email notifications.

Credentials come from .env:
    SMTP_USER   — your Gmail address
    SMTP_PASS   — 16-char Google App Password (NOT your account password)
    NOTIFY_EMAIL — where to send notifications (usually same as SMTP_USER)
    APP_BASE_URL — e.g. http://localhost:8000

When SMTP_USER / SMTP_PASS are unset (or left as dummy values) the functions
print what they would send instead of crashing — safe for local dev without
a real Gmail App Password configured yet.
"""

import html
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

_DUMMY_MARKER = "dummy"


def _cfg():
    return {
        "user":     os.getenv("SMTP_USER", ""),
        "password": os.getenv("SMTP_PASS", ""),
        "notify":   os.getenv("NOTIFY_EMAIL", os.getenv("SMTP_USER", "")),
        "base_url": os.getenv("APP_BASE_URL", "http://localhost:8000"),
    }


def _is_configured(cfg: dict) -> bool:
    return bool(cfg["user"] and cfg["password"] and _DUMMY_MARKER not in cfg["password"])


def _send(subject: str, html_body: str) -> None:
    cfg = _cfg()
    if not _is_configured(cfg):
        print(f"\n[emailer] SMTP not configured — would send:\nSubject: {subject}\n{html_body[:400]}\n")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = cfg["user"]
    msg["To"]      = cfg["notify"]
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(cfg["user"], cfg["password"])
        server.sendmail(cfg["user"], cfg["notify"], msg.as_string())


def _base_template(title: str, body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body   {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
              background:#080810; color:#D0CAEB; margin:0; padding:0; }}
    .wrap  {{ max-width:680px; margin:40px auto; padding:32px;
              background:#0F0F1A; border:1px solid rgba(255,255,255,.08);
              border-radius:16px; }}
    h1     {{ font-size:1.3rem; color:#F2EEFF; margin:0 0 8px; }}
    .sub   {{ font-size:.8rem; color:#7B769A; margin-bottom:24px; }}
    .prose {{ font-size:.9rem; line-height:1.75; white-space:pre-wrap;
              background:#05050C; border:1px solid rgba(255,255,255,.06);
              border-radius:8px; padding:20px; color:#C8C3E2; }}
    .btn   {{ display:inline-block; margin:6px 8px 6px 0; padding:10px 22px;
              border-radius:8px; font-size:.85rem; font-weight:600;
              text-decoration:none; }}
    .btn-approve {{ background:#22D473; color:#080810; }}
    .btn-skip    {{ background:#EF4444; color:#fff; }}
    .btn-revise  {{ background:#7C3AED; color:#fff; }}
    .footer {{ font-size:.72rem; color:#4A4568; margin-top:28px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{html.escape(title)}</h1>
    {body_html}
    <div class="footer">Blog Agent · {html.escape(_cfg()["base_url"])}</div>
  </div>
</body>
</html>"""


# ── Public API ────────────────────────────────────────────────────────────────

def send_review_email(post_id: str, topic: str, draft: str) -> None:
    """
    Sent at 8:30 AM after a draft is generated.
    Contains the full draft + approve / revise / skip action buttons.
    """
    cfg = _cfg()
    review_url = f"{cfg['base_url']}/review/{post_id}"

    body = f"""
    <div class="sub">Blog draft ready for review</div>
    <p><strong>Topic:</strong> {html.escape(topic)}</p>
    <p>
      <a href="{html.escape(review_url)}/approve" class="btn btn-approve">Approve</a>
      <a href="{html.escape(review_url)}" class="btn btn-revise">Review &amp; Revise</a>
      <a href="{html.escape(review_url)}/skip" class="btn btn-skip">Skip</a>
    </p>
    <div class="prose">{html.escape(draft)}</div>
    <p><a href="{html.escape(review_url)}" style="color:#A78BFA;">Open full review page →</a></p>
    """
    _send(f"[Blog Agent] Review: {topic}", _base_template(f"Review: {topic}", body))


def send_published_email(post_id: str, topic: str, linkedin_url: str, medium_md: str) -> None:
    """
    Sent at 9:00 AM after a post is published to LinkedIn.
    Includes Medium paste-ready markdown + new-story link.
    """
    medium_link = "https://medium.com/new-story"
    body = f"""
    <div class="sub">Post published successfully</div>
    <p><strong>Topic:</strong> {html.escape(topic)}</p>
    {"<p><strong>LinkedIn:</strong> <a href='" + html.escape(linkedin_url) + "' style='color:#A78BFA;'>View post →</a></p>" if linkedin_url else "<p><em>LinkedIn: see logs for result</em></p>"}
    <p><strong>Medium:</strong>
       <a href="{html.escape(medium_link)}" class="btn btn-revise" style="font-size:.78rem;padding:7px 14px;">
         Paste to Medium →
       </a>
    </p>
    <p style="font-size:.78rem;color:#7B769A;">Copy the markdown below and paste it into Medium's editor:</p>
    <div class="prose">{html.escape(medium_md[:3000])}</div>
    """
    _send(f"[Blog Agent] Published: {topic}", _base_template(f"Published: {topic}", body))


def send_skipped_email(post_id: str, topic: str) -> None:
    """Sent when a draft was never approved before the publish window."""
    cfg = _cfg()
    review_url = f"{cfg['base_url']}/review/{post_id}"
    body = f"""
    <div class="sub">Draft skipped — not published</div>
    <p><strong>Topic:</strong> {html.escape(topic)}</p>
    <p>This draft was not approved before the publish window. It is still available if you change your mind:</p>
    <p><a href="{html.escape(review_url)}" class="btn btn-revise">Open draft →</a></p>
    """
    _send(f"[Blog Agent] Skipped: {topic}", _base_template(f"Skipped: {topic}", body))


def send_error_email(subject: str, message: str) -> None:
    """Generic error notification — e.g. LinkedIn token expired."""
    body = f"""
    <div class="sub">Action required</div>
    <div class="prose">{html.escape(message)}</div>
    """
    _send(f"[Blog Agent] {subject}", _base_template(subject, body))
