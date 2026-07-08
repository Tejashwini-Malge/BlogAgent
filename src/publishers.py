"""
Publishing layer.

LinkedIn  — real API (POST /v2/ugcPosts, Bearer token, 2900-char limit).
Medium    — manual only (API closed to new integrations since Jan 2025).
revise_post — single OpenRouter LLM call to apply corrections.

All integrations degrade gracefully when credentials are missing/dummy.
"""

import html
import json
import os
import re
import textwrap

import requests

_DUMMY = "dummy"


def _li_configured() -> bool:
    token = os.getenv("LINKEDIN_ACCESS_TOKEN", "")
    return bool(token and _DUMMY not in token)


def markdown_to_linkedin(md: str) -> str:
    """
    Convert Markdown → plain text suitable for a LinkedIn post.
    - Strip H1/H2/H3 markers (keep the text)
    - Remove bold/italic markers
    - Remove inline code ticks
    - Replace Markdown bullets with •
    - Collapse excessive blank lines
    """
    text = md
    text = re.sub(r"^#{1,3}\s+", "", text, flags=re.MULTILINE)   # headings
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)                  # bold
    text = re.sub(r"\*(.+?)\*",   r"\1", text)                    # italic
    text = re.sub(r"`(.+?)`",     r"\1", text)                    # inline code
    text = re.sub(r"```[\s\S]*?```", "", text)                     # code blocks
    text = re.sub(r"^[-*+]\s+", "• ", text, flags=re.MULTILINE)   # bullets
    text = re.sub(r"\n{3,}", "\n\n", text)                        # blank lines
    return text.strip()


def publish_linkedin(text: str) -> dict:
    """
    Post `text` to LinkedIn. Truncates at 2900 chars.
    Returns {"success": bool, "url": str | None, "error": str | None}.
    401 is surfaced as a loud actionable error (not silently retried).
    """
    if not _li_configured():
        print("[publishers] LinkedIn not configured — skipping post (dummy credentials).")
        return {"success": False, "url": None, "error": "LinkedIn not configured"}

    token      = os.getenv("LINKEDIN_ACCESS_TOKEN")
    person_urn = os.getenv("LINKEDIN_PERSON_URN", "")

    # Hard LinkedIn character limit
    if len(text) > 2900:
        text = text[:2897] + "…"

    payload = {
        "author": person_urn,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": text},
                "shareMediaCategory": "NONE",
            }
        },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
    }

    try:
        resp = requests.post(
            "https://api.linkedin.com/v2/ugcPosts",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "X-Restli-Protocol-Version": "2.0.0",
            },
            json=payload,
            timeout=30,
        )
    except requests.RequestException as exc:
        return {"success": False, "url": None, "error": str(exc)}

    if resp.status_code == 401:
        msg = "LinkedIn token expired — re-run OAuth to get a new 60-day token."
        print(f"[publishers] {msg}")
        return {"success": False, "url": None, "error": msg}

    if resp.status_code not in (200, 201):
        return {
            "success": False,
            "url": None,
            "error": f"LinkedIn API error {resp.status_code}: {resp.text[:200]}",
        }

    post_id = resp.headers.get("x-restli-id", "")
    url = f"https://www.linkedin.com/feed/update/{post_id}/" if post_id else None
    return {"success": True, "url": url, "error": None}


def publish_medium(markdown: str, title: str = "") -> dict:
    """
    Medium's API is closed to new integrations since 1 Jan 2025.
    Only pre-2025 legacy tokens still work; new tokens are NOT issued.

    This function:
    - Returns the markdown ready to paste into https://medium.com/new-story
    - Never browser-automates (account-risk on a personal brand)
    - Uses a legacy token ONLY if MEDIUM_TOKEN is set AND not a dummy value
    """
    token = os.getenv("MEDIUM_TOKEN", "")

    if token and _DUMMY not in token:
        # Try the legacy REST API (may work if you have a pre-2025 token)
        try:
            me = requests.get(
                "https://api.medium.com/v1/me",
                headers={"Authorization": f"Bearer {token}"},
                timeout=15,
            )
            if me.status_code == 200:
                user_id = me.json()["data"]["id"]
                resp = requests.post(
                    f"https://api.medium.com/v1/users/{user_id}/posts",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "title": title or "Blog Post",
                        "contentFormat": "markdown",
                        "content": markdown,
                        "publishStatus": "draft",
                    },
                    timeout=30,
                )
                if resp.status_code in (200, 201):
                    url = resp.json().get("data", {}).get("url", "")
                    return {"success": True, "url": url, "manual_paste": False}
        except requests.RequestException:
            pass

    # Default: manual paste flow
    return {
        "success": True,
        "url": "https://medium.com/new-story",
        "manual_paste": True,
        "markdown": markdown,
    }


def revise_post(draft: str, corrections: str) -> str:
    """
    Call OpenRouter with a focused prompt: apply ONLY the specified corrections,
    preserve voice, return full markdown only.

    Falls back to the original draft if the API call fails.
    """
    import os
    try:
        from openai import OpenAI
    except ImportError:
        print("[publishers] openai package not available — returning original draft.")
        return draft

    api_key        = os.getenv("OPENAI_API_KEY", "")
    api_base       = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    model          = os.getenv("OPENAI_MODEL_NAME", "openai/gpt-4o-mini")
    fallback_model = os.getenv("OPENAI_FALLBACK_MODEL_NAME", "llama-3.1-8b-instant")

    if not api_key or _DUMMY in api_key:
        print("[publishers] OpenRouter not configured — returning original draft.")
        return draft

    client = OpenAI(api_key=api_key, base_url=api_base)

    system = textwrap.dedent("""
        You are an editorial assistant. Your ONLY job is to apply the specific
        corrections listed by the author to the blog post draft.

        Rules:
        - Apply ONLY the corrections described. Do not add new content.
        - Preserve the author's voice, tone, and phrasing everywhere else.
        - Return the complete revised post in Markdown. Nothing else.
        - Do not add explanations, commentary, or change headings unless asked.
    """).strip()

    user_msg = f"""CORRECTIONS TO APPLY:
{corrections}

ORIGINAL DRAFT:
{draft}"""

    def _call(use_model: str) -> str:
        resp = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=3000,
        )
        return (resp.choices[0].message.content or "").strip()

    try:
        revised = _call(model)
        return revised if revised else draft
    except Exception as exc:
        # Groq quotas are per-model — a smaller fallback model may still have budget
        if fallback_model and ("429" in str(exc) or "rate_limit" in str(exc)):
            print(f"[publishers] {model} rate-limited; retrying on {fallback_model}")
            try:
                revised = _call(fallback_model)
                return revised if revised else draft
            except Exception as exc2:
                exc = exc2
        print(f"[publishers] revise_post failed: {exc} — returning original draft.")
        return draft
