"""
Signup / login / logout / profile — self-contained HTML pages, same pattern
as app.py's /review/{post_id}: a FastAPI route returning a complete
HTMLResponse, not part of the frontend/ SPA's client-side JS.

Unlike /review/{post_id} (which hardcodes its own hex colors), these link
directly to /static/style.css so they reuse the app's actual design tokens
and component classes (.form-card, .form-input, .form-label, .btn-generate)
rather than reinventing them.
"""
import html

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from src import accounts
from src.auth import SESSION_COOKIE, require_user_page

auth_router = APIRouter()

_SESSION_COOKIE_MAX_AGE = 30 * 24 * 60 * 60  # 30 days, matches accounts._SESSION_LIFETIME


def _page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} — Blog Agent</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Special+Elite&family=Courier+Prime:ital,wght@0,400;0,700;1,400&family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/static/style.css?v=8">
  <style>
    body {{ display: flex; align-items: center; justify-content: center; min-height: 100vh; }}
    .auth-wrap {{ width: 100%; max-width: 420px; padding: 1.5rem; }}
    .auth-wrap .form-card {{ transform: none; }}
    .auth-card__inner {{ padding: 1.75rem 1.5rem; }}
    .auth-title {{ font-family: 'Special Elite', serif; font-size: 1.3rem; color: var(--text); margin-bottom: .3rem; }}
    .auth-sub {{ font-size: .82rem; color: var(--paper-faint); margin-bottom: 1.4rem; }}
    .auth-field {{ margin-bottom: 1rem; }}
    .auth-field .form-input {{ width: 100%; }}
    .auth-field .form-label {{ display: block; margin-bottom: .4rem; }}
    .auth-submit {{ width: 100%; justify-content: center; margin-top: .3rem; }}
    .auth-error {{ background: var(--red-tint); border: 1px solid var(--red); color: var(--red);
                   border-radius: var(--r-sm); padding: .6rem .8rem; font-size: .82rem; margin-bottom: 1rem; }}
    .auth-switch {{ margin-top: 1.1rem; font-size: .82rem; color: var(--paper-faint); text-align: center; }}
    .auth-switch a {{ color: var(--teal-b); text-decoration: none; }}
    .auth-switch a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div class="auth-wrap">
    <div class="form-card">
      <div class="tw-slot__inner auth-card__inner">
        {body}
      </div>
    </div>
  </div>
</body>
</html>"""


def _signup_body(error: str = "", email: str = "") -> str:
    error_html = f'<div class="auth-error">{html.escape(error)}</div>' if error else ""
    return f"""
    <div class="auth-title">Create your account</div>
    <div class="auth-sub">Your blog posts and topic queue stay attached to your own profile.</div>
    {error_html}
    <form method="post" action="/signup">
      <div class="auth-field">
        <label class="form-label" for="email">Email</label>
        <input class="form-input" type="email" id="email" name="email" value="{html.escape(email)}"
               placeholder="you@example.com" required autofocus>
      </div>
      <div class="auth-field">
        <label class="form-label" for="password">Password</label>
        <input class="form-input" type="password" id="password" name="password"
               placeholder="At least 8 characters" required minlength="8">
      </div>
      <div class="auth-field">
        <label class="form-label" for="confirm">Confirm password</label>
        <input class="form-input" type="password" id="confirm" name="confirm"
               placeholder="Same password again" required minlength="8">
      </div>
      <button class="btn-generate auth-submit" type="submit">Sign up</button>
    </form>
    <div class="auth-switch">Already have an account? <a href="/login">Log in</a></div>
    """


def _login_body(error: str = "", email: str = "") -> str:
    error_html = f'<div class="auth-error">{html.escape(error)}</div>' if error else ""
    return f"""
    <div class="auth-title">Welcome back</div>
    <div class="auth-sub">Log in to your Blog Agent profile.</div>
    {error_html}
    <form method="post" action="/login">
      <div class="auth-field">
        <label class="form-label" for="email">Email</label>
        <input class="form-input" type="email" id="email" name="email" value="{html.escape(email)}"
               placeholder="you@example.com" required autofocus>
      </div>
      <div class="auth-field">
        <label class="form-label" for="password">Password</label>
        <input class="form-input" type="password" id="password" name="password"
               placeholder="Your password" required>
      </div>
      <button class="btn-generate auth-submit" type="submit">Log in</button>
    </form>
    <div class="auth-switch">Don't have an account? <a href="/signup">Sign up</a></div>
    """


def _profile_body(user: dict, post_count: int) -> str:
    member_since = (user.get("created_at") or "")[:10]
    return f"""
    <div class="auth-title">Your profile</div>
    <div class="auth-sub">{html.escape(user["email"])}</div>
    <div class="auth-field">
      <label class="form-label">Member since</label>
      <div class="form-input" style="display:flex;align-items:center;">{html.escape(member_since)}</div>
    </div>
    <div class="auth-field">
      <label class="form-label">Posts generated</label>
      <div class="form-input" style="display:flex;align-items:center;">{post_count}</div>
    </div>
    <form method="post" action="/logout">
      <button class="btn-generate auth-submit" type="submit">Log out</button>
    </form>
    """


def _set_session_cookie(response: RedirectResponse, token: str) -> RedirectResponse:
    response.set_cookie(
        SESSION_COOKIE, token, max_age=_SESSION_COOKIE_MAX_AGE,
        httponly=True, samesite="lax",
    )
    return response


@auth_router.get("/signup", response_class=HTMLResponse)
async def signup_page():
    return HTMLResponse(_page("Sign up", _signup_body()))


@auth_router.post("/signup")
async def signup_submit(email: str = Form(...), password: str = Form(...), confirm: str = Form(...)):
    if password != confirm:
        return HTMLResponse(_page("Sign up", _signup_body("Passwords don't match.", email)))
    try:
        user = accounts.create_user(email, password)
    except accounts.AccountError as exc:
        return HTMLResponse(_page("Sign up", _signup_body(str(exc), email)))

    token = accounts.create_session(user["id"])
    return _set_session_cookie(RedirectResponse(url="/", status_code=303), token)


@auth_router.get("/login", response_class=HTMLResponse)
async def login_page():
    return HTMLResponse(_page("Log in", _login_body()))


@auth_router.post("/login")
async def login_submit(email: str = Form(...), password: str = Form(...)):
    try:
        user = accounts.authenticate(email, password)
    except accounts.AccountError as exc:
        return HTMLResponse(_page("Log in", _login_body(str(exc), email)))

    token = accounts.create_session(user["id"])
    return _set_session_cookie(RedirectResponse(url="/", status_code=303), token)


@auth_router.post("/logout")
async def logout(request: Request):
    token = request.cookies.get(SESSION_COOKIE, "")
    if token:
        accounts.delete_session(token)
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    return response


@auth_router.get("/profile", response_class=HTMLResponse)
async def profile_page(user: dict = Depends(require_user_page)):
    from src import pending as pending_store

    post_count = len(pending_store.posts_for_user(user["id"]))
    return HTMLResponse(_page("Profile", _profile_body(user, post_count)))
