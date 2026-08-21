# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Single-URL Agent Engine Optimization (AEO) analyzer on AWS Lambda durable
functions (Python). Deterministic scoring, reproducible results.

AEO = making a site discoverable, parsable, and usable by autonomous AI agents.
Scores a page against the five-signal model from Addy Osmani's AEO framework
(Google Cloud AI, April 2026): Discoverability, Parsability, Token Efficiency,
Capability Signaling, Access Control.

Scoring is fully deterministic (no LLM involvement). Every point maps to a
measurable, threshold-based check. Bedrock is used only for generating prose
analysis explaining the scores and recommending fixes.

Pipeline (durable workflow):
  1. fetch_and_parse   (step)     GET html + robots.txt + llms.txt + AGENTS.md +
                                  skill.md + markdown twin, extract AEO signals
  2. score_signals     (step)     Deterministic scoring against published rubric
  3. analyze_with_ai   (step)     Bedrock generates prose findings (temp=0)
  4. publish_report    (step)     Render HTML + JSON, write to S3
  5. record_outcome    (step)     Write result to DynamoDB

Optional: set APPROVAL_GATE=true to enable a human review callback between
steps 3 and 4. Default runs all steps without stopping.

Grounding:
  - Osmani AEO framework (Google Cloud AI, Apr 2026): five agent-readiness signals
  - Osmani agentic-seo tool: published weights (25/25/25/15/10), deterministic checks
  - RFC 9309 robots.txt: AI-agent crawler access
  - llms.txt (proposed standard, llmstxt.org): agent-readable index at root
  - AGENTS.md / skill.md: agent capability signaling

SDK: aws-durable-execution-sdk-python (Python >=3.11).
"""

import os
import re
import json
import urllib.request
import urllib.error
import html as _html
from urllib.parse import urlparse

import boto3

from aws_durable_execution_sdk_python import (
    DurableContext,
    StepContext,
    durable_execution,
    durable_step,
)
from aws_durable_execution_sdk_python.config import StepConfig

ARTIFACTS_BUCKET = os.environ["ARTIFACTS_BUCKET"]
STATE_TABLE = os.environ["STATE_TABLE"]
BEDROCK_MODEL_ID = os.environ["BEDROCK_MODEL_ID"]
APPROVAL_GATE = os.environ.get("APPROVAL_GATE", "false").lower() == "true"
BEDROCK_TEMPERATURE = float(os.environ.get("BEDROCK_TEMPERATURE", "0"))

s3 = boto3.client("s3")
ddb = boto3.client("dynamodb")
bedrock = boto3.client("bedrock-runtime")

# Fetch as an AI agent would - ClaudeBot UA for realistic testing
UA = "ClaudeBot/1.0 (AEO-Analyzer; +https://github.com/aws-samples/aeo-scoring-pipeline)"
AI_BOTS = ["GPTBot", "PerplexityBot", "ClaudeBot", "Claude-Web", "Google-Extended", "CCBot"]

# Osmani token thresholds
TOKEN_CEILING = 30000
TOKEN_QUICKSTART = 15000
LLMS_TXT_TOKEN_LIMIT = 5000  # ~20KB at ~4 chars/token


def _http_get(url: str, timeout: int = 15) -> tuple:
    """Return (status, text). Never raises for HTTP errors; returns status."""
    if not url.startswith("https://"):
        raise ValueError(f"Only https:// URLs are permitted, got: {url[:40]}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(2_000_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception:
        return 0, ""


# ---------------------------------------------------------------------------
# Step 1: Fetch and extract signals
# ---------------------------------------------------------------------------

@durable_step
def fetch_and_parse(step_context: StepContext, url: str) -> dict:
    """Fetch the page + supporting files and extract structured signals.

    Durable step output is capped at 262144 bytes. We parse in-place and
    return only the small structured signals dict.
    """
    parsed = urlparse(url)
    root = f"{parsed.scheme}://{parsed.netloc}"

    status, html = _http_get(url)
    step_context.logger.info(f"fetched {url} -> {status}, {len(html)} bytes")

    _, robots = _http_get(f"{root}/robots.txt")
    llms_status, llms_body = _http_get(f"{root}/llms.txt")
    agents_status, _ = _http_get(f"{root}/AGENTS.md")
    skill_status, _ = _http_get(f"{root}/skill.md")
    md_status, _ = _http_get(url.rstrip("/") + ".md")

    return _extract_signals(
        url, status, html, robots,
        has_llms=(llms_status == 200),
        llms_bytes=len(llms_body.encode("utf-8")) if llms_body else 0,
        has_agents_md=(agents_status == 200),
        has_skill_md=(skill_status == 200),
        has_md_twin=(md_status == 200),
    )


def _extract_signals(url: str, http_status: int, html: str, robots: str,
                     has_llms: bool, llms_bytes: int, has_agents_md: bool,
                     has_skill_md: bool, has_md_twin: bool) -> dict:
    """Extract measurable AEO signals from fetched content."""

    # Headings
    headings = re.findall(r"<h([1-6])[^>]*>(.*?)</h\1>", html, re.I | re.S)
    heading_levels = [int(h[0]) for h in headings]
    heading_texts = [re.sub(r"<[^>]+>", "", h[1]).strip() for h in headings]
    heading_texts = [h for h in heading_texts if h]
    has_h1 = 1 in heading_levels

    # Check heading hierarchy (no skipping levels)
    hierarchy_valid = True
    for i in range(1, len(heading_levels)):
        if heading_levels[i] > heading_levels[i-1] + 1:
            hierarchy_valid = False
            break

    # Code blocks and tables
    fenced_code = len(re.findall(r"<pre[^>]*>|<code[^>]*>", html, re.I))
    real_tables = len(re.findall(r"<table[^>]*>", html, re.I))

    # Body text extraction
    body_text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.I | re.S)
    body_text = re.sub(r"<[^>]+>", " ", body_text)
    body_text = re.sub(r"\s+", " ", body_text).strip()
    word_count = len(body_text.split())
    est_tokens = int(word_count * 1.3)

    # JS-heavy detection
    script_tags = len(re.findall(r"<script", html, re.I))
    likely_client_rendered = word_count < 200 and script_tags >= 3

    # HTML noise ratio (non-content bytes vs total)
    html_bytes = len(html.encode("utf-8"))
    text_bytes = len(body_text.encode("utf-8"))
    noise_ratio = 1 - (text_bytes / html_bytes) if html_bytes > 0 else 1

    # Robots.txt analysis
    blocked_bots = []
    allowed_bots = []
    for bot in AI_BOTS:
        pat = re.compile(rf"user-agent:\s*{re.escape(bot)}(.*?)(?=user-agent:|$)", re.I | re.S)
        m = pat.search(robots)
        if m:
            section = m.group(1)
            if re.search(r"disallow:\s*/\s*($|\n)", section, re.I):
                blocked_bots.append(bot)
            elif re.search(r"allow:\s*/", section, re.I):
                allowed_bots.append(bot)

    has_ai_rules = bool(re.search(
        r"gptbot|claudebot|perplexitybot|google-extended|ccbot",
        robots, re.I
    ))

    # Sitemap reference in robots.txt
    has_sitemap = bool(re.search(r"sitemap:", robots, re.I))

    return {
        "url": url,
        "http_status": http_status,
        # Discoverability
        "server_rendered": word_count >= 200,
        "likely_client_rendered": likely_client_rendered,
        "has_llms_txt": has_llms,
        "llms_txt_bytes": llms_bytes,
        "llms_txt_est_tokens": int(llms_bytes / 4),  # ~4 chars/token
        "ai_bots_blocked": blocked_bots,
        "ai_bots_allowed": allowed_bots,
        "has_sitemap": has_sitemap,
        "has_agents_md": has_agents_md,
        # Parsability
        "has_h1": has_h1,
        "heading_count": len(heading_texts),
        "heading_levels": heading_levels[:20],
        "hierarchy_valid": hierarchy_valid,
        "sample_headings": heading_texts[:10],
        "has_fenced_code": fenced_code > 0,
        "code_block_count": fenced_code,
        "table_count": real_tables,
        "has_markdown_twin": has_md_twin,
        "noise_ratio": round(noise_ratio, 3),
        # Token Efficiency
        "estimated_tokens": est_tokens,
        "word_count": word_count,
        "within_token_ceiling": est_tokens <= TOKEN_CEILING,
        "within_quickstart_target": est_tokens <= TOKEN_QUICKSTART,
        # Capability Signaling
        "has_skill_md": has_skill_md,
        # Access Control
        "robots_has_ai_rules": has_ai_rules,
        "robots_blocked_bots": blocked_bots,
        "robots_allowed_bots": allowed_bots,
    }


# ---------------------------------------------------------------------------
# Step 2: Deterministic scoring (no LLM, fully reproducible)
# ---------------------------------------------------------------------------
# Weights aligned with Osmani's agentic-seo tool: 25/25/25/15/10 = 100
# Every point maps to a measurable condition.

@durable_step
def score_signals(step_context: StepContext, signals: dict) -> dict:
    """Score signals deterministically. Same input = same output, always."""

    scores = {}

    # --- DISCOVERABILITY (25 points) ---
    disc = 0
    # HTTP 200 with server-rendered content (8 pts)
    if signals["http_status"] == 200 and signals["server_rendered"]:
        disc += 8
    elif signals["http_status"] == 200:
        disc += 4  # page loads but thin content (likely JS-rendered)

    # robots.txt does not block AI crawlers (7 pts)
    if not signals["ai_bots_blocked"]:
        disc += 7
    elif len(signals["ai_bots_blocked"]) < len(AI_BOTS):
        disc += 3  # some blocked, not all

    # llms.txt exists and is reasonably sized (7 pts)
    if signals["has_llms_txt"]:
        if signals["llms_txt_est_tokens"] <= LLMS_TXT_TOKEN_LIMIT:
            disc += 7  # exists and concise
        elif signals["llms_txt_est_tokens"] <= LLMS_TXT_TOKEN_LIMIT * 3:
            disc += 4  # exists but oversized
        else:
            disc += 2  # exists but bloated (>15K tokens)
    # else: 0

    # AGENTS.md at domain root (3 pts)
    if signals["has_agents_md"]:
        disc += 3

    scores["discoverability"] = {"score": disc, "max": 25}

    # --- PARSABILITY (25 points) ---
    pars = 0
    # H1 present (3 pts)
    if signals["has_h1"]:
        pars += 3

    # Heading hierarchy - count and validity (7 pts)
    hcount = signals["heading_count"]
    if hcount >= 5 and signals["hierarchy_valid"]:
        pars += 7
    elif hcount >= 3:
        pars += 5
    elif hcount >= 1:
        pars += 2

    # Code blocks present (4 pts)
    if signals["code_block_count"] >= 3:
        pars += 4
    elif signals["has_fenced_code"]:
        pars += 2

    # Tables present (3 pts)
    if signals["table_count"] >= 1:
        pars += 3

    # Markdown twin available (4 pts)
    if signals["has_markdown_twin"]:
        pars += 4

    # Low HTML noise ratio (4 pts) - content > 50% of page bytes
    if signals["noise_ratio"] <= 0.5:
        pars += 4
    elif signals["noise_ratio"] <= 0.7:
        pars += 2

    scores["parsability"] = {"score": pars, "max": 25}

    # --- TOKEN EFFICIENCY (25 points) ---
    toke = 0
    tokens = signals["estimated_tokens"]

    if signals["http_status"] != 200 or not signals["server_rendered"]:
        # Can't evaluate if no content retrieved
        toke = 0
    elif tokens <= TOKEN_QUICKSTART:
        toke += 20  # well under budget
    elif tokens <= TOKEN_CEILING:
        toke += 14  # within ceiling but over quickstart
    elif tokens <= TOKEN_CEILING * 2:
        toke += 7   # oversized but not catastrophic
    # else: 0 (>60K tokens)

    # Not JS-dependent (5 pts)
    if not signals["likely_client_rendered"]:
        toke += 5

    scores["token_efficiency"] = {"score": toke, "max": 25}

    # --- CAPABILITY SIGNALING (15 points) ---
    caps = 0
    # skill.md exists (8 pts)
    if signals["has_skill_md"]:
        caps += 8

    # AGENTS.md exists (already counted in discoverability, but also signals capability)
    if signals["has_agents_md"]:
        caps += 5

    # Structured content signals capability (code examples = API usage patterns) (2 pts)
    if signals["code_block_count"] >= 2 and signals["table_count"] >= 1:
        caps += 2

    scores["capability_signaling"] = {"score": caps, "max": 15}

    # --- ACCESS CONTROL (10 points) ---
    accs = 0
    # Explicit AI rules in robots.txt (6 pts)
    if signals["robots_has_ai_rules"]:
        if signals["robots_allowed_bots"]:
            accs += 6  # explicit allow = intentional access
        elif signals["robots_blocked_bots"]:
            accs += 4  # explicit block = intentional boundary (still good practice)
        else:
            accs += 3  # mentioned but unclear intent

    # Sitemap declared (2 pts)
    if signals["has_sitemap"]:
        accs += 2

    # Not blocking AND has intentional rules (2 pts bonus)
    if not signals["ai_bots_blocked"] and signals["robots_has_ai_rules"]:
        accs += 2

    scores["access_control"] = {"score": accs, "max": 10}

    # --- Overall ---
    total = sum(s["score"] for s in scores.values())
    total_max = sum(s["max"] for s in scores.values())

    result = {
        "url": signals["url"],
        "overall_score": total,
        "max_score": total_max,
        "signals": signals,
        "scores": scores,
    }

    step_context.logger.info(f"scored {signals['url']} -> {total}/{total_max}")
    return result


# ---------------------------------------------------------------------------
# Step 3: Bedrock analysis (prose findings + recommendations, temp=0)
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = """You are an AEO (Agent Engine Optimization) analyst. You receive
a page's deterministic AEO scores and raw signals. Your job is to explain the scores
in clear prose and recommend specific fixes.

The scores are already computed deterministically. Do NOT reassign or override them.
Your role is to:
1. Explain WHY each signal scored the way it did (cite specific evidence from signals)
2. Recommend the top 3 highest-impact fixes, each tied to a specific signal
3. Provide a 2-3 sentence overall summary

Return valid JSON with this structure:
{
  "summary": "2-3 sentence plain assessment",
  "findings": [
    {"area": "discoverability|parsability|token_efficiency|capability_signaling|access_control",
     "score": "<actual>/<max>",
     "status": "pass|warn|fail",
     "detail": "explanation of what was found",
     "fix": "specific action to improve (null if pass)"}
  ],
  "top_fixes": ["fix 1", "fix 2", "fix 3"]
}

Set status based on score percentage: >=80% of max = pass, 50-79% = warn, <50% = fail.
Return ONLY valid JSON."""


@durable_step
def analyze_with_ai(step_context: StepContext, scored: dict) -> dict:
    """Generate prose analysis of the deterministic scores."""
    prompt = f"{ANALYSIS_PROMPT}\n\nScored result:\n{json.dumps(scored, indent=2)}"
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1500,
        "temperature": BEDROCK_TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = bedrock.invoke_model(modelId=BEDROCK_MODEL_ID, body=json.dumps(body))
    payload = json.loads(resp["body"].read())
    text = payload.get("content", [{}])[0].get("text", "").strip()

    try:
        analysis = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        analysis = json.loads(m.group(0)) if m else {"raw": text}

    # Merge deterministic scores into the analysis output
    analysis["overall_score"] = scored["overall_score"]
    analysis["max_score"] = scored["max_score"]
    analysis["scores"] = scored["scores"]

    return {"url": scored["url"], "report": analysis}


# ---------------------------------------------------------------------------
# HTML report renderer
# ---------------------------------------------------------------------------

def render_aeo_html(url: str, report: dict) -> str:
    def esc(s):
        return _html.escape(str(s))

    score = report.get("overall_score", 0)
    max_score = report.get("max_score", 100)
    summary = report.get("summary", "")
    findings = report.get("findings", [])
    fixes = report.get("top_fixes", [])
    scores = report.get("scores", {})

    status_style = {
        "pass": ("PASS", "#e8edf5", "#2d3f63", "#4a6fa5"),
        "warn": ("REVIEW", "#f5efe0", "#5c4d2e", "#a5894a"),
        "fail": ("GAP", "#efe9f2", "#4a3a54", "#7a5a8a"),
    }

    try:
        s = float(score)
    except (TypeError, ValueError):
        s = 0
    gauge = "#3a5a8c" if s >= 75 else ("#5a6a8c" if s >= 50 else "#8a7a9a")

    # Score breakdown bar
    breakdown = ""
    for area, data in scores.items():
        pct = int((data["score"] / data["max"]) * 100) if data["max"] > 0 else 0
        breakdown += (
            '<div class="sbar"><span class="slabel">%s</span>'
            '<div class="strack"><div class="sfill" style="width:%d%%"></div></div>'
            '<span class="snum">%d/%d</span></div>'
        ) % (esc(area.replace("_", " ").title()), pct, data["score"], data["max"])

    rows = ""
    for f in findings:
        st = (f.get("status") or "warn").lower()
        label, bg, fg, bar = status_style.get(st, status_style["warn"])
        fix_html = f'<p class="fix">Fix: {esc(f.get("fix", ""))}</p>' if f.get("fix") else ""
        rows += (
            '<div class="finding"><div class="fbar" style="background:%s"></div>'
            '<div class="fbody"><div class="fhead"><span class="area">%s</span>'
            '<span class="fscore">%s</span>'
            '<span class="badge" style="background:%s;color:%s">%s</span></div>'
            '<p class="detail">%s</p>%s</div></div>'
        ) % (bar, esc(f.get("area", "").replace("_", " ").title()),
             esc(f.get("score", "")), bg, fg, label,
             esc(f.get("detail", "")), fix_html)

    fix_items = "".join("<li>%s</li>" % esc(x) for x in fixes) or "<li>No priority fixes identified.</li>"

    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        '<title>AEO Report - %s</title><style>'
        'body{margin:0;background:#f5f6f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;color:#1f2734;line-height:1.55}'
        '.wrap{max-width:840px;margin:0 auto;padding:32px 24px}'
        'header{background:linear-gradient(135deg,#2d3f63,#3a5a8c);color:#fff;border-radius:14px;padding:28px 30px}'
        'header .kicker{font-size:12px;letter-spacing:.12em;text-transform:uppercase;opacity:.8;margin:0 0 6px}'
        'header h1{margin:0;font-size:20px;word-break:break-all}'
        '.scorewrap{display:flex;align-items:center;gap:26px;margin-top:22px}'
        '.gauge{width:104px;height:104px;border-radius:50%%;flex-shrink:0;background:conic-gradient(%s calc(%s*3.6deg),rgba(255,255,255,.22) 0);display:grid;place-items:center}'
        '.gauge .inner{width:82px;height:82px;border-radius:50%%;background:#2d3f63;display:grid;place-items:center;text-align:center}'
        '.gauge .num{font-size:28px;font-weight:700;color:#fff}.gauge .lbl{font-size:10px;letter-spacing:.1em;opacity:.8;color:#fff}'
        '.summary{margin:0;font-size:15px;opacity:.95}'
        '.breakdown{margin-top:18px}'
        '.sbar{display:flex;align-items:center;gap:10px;margin:6px 0}'
        '.slabel{font-size:12px;width:140px;opacity:.85;color:#fff}'
        '.strack{flex:1;height:6px;background:rgba(255,255,255,.2);border-radius:3px}'
        '.sfill{height:100%%;background:#7ab3ff;border-radius:3px}'
        '.snum{font-size:12px;opacity:.8;width:40px;text-align:right;color:#fff}'
        'section{background:#fff;border:1px solid #e6e9ef;border-radius:12px;padding:22px 24px;margin-top:18px}'
        'section h2{margin:0 0 16px;font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#8a94a6}'
        '.finding{display:flex;gap:14px;padding:14px 0;border-top:1px solid #eef0f4}.finding:first-of-type{border-top:none}'
        '.fbar{width:4px;border-radius:4px;flex-shrink:0}'
        '.fhead{display:flex;align-items:center;gap:12px}'
        '.area{font-weight:600;font-size:15px}'
        '.fscore{font-size:13px;color:#5a6a8c}'
        '.badge{font-size:11px;font-weight:700;letter-spacing:.06em;padding:3px 10px;border-radius:100px}'
        '.detail{margin:6px 0 4px;font-size:14px;color:#4a5568}'
        '.fix{margin:2px 0 0;font-size:13px;color:#3a5a8c;font-style:italic}'
        'ol{margin:0;padding-left:20px}ol li{margin:8px 0;font-size:14px}'
        'footer{margin-top:22px;font-size:12px;color:#8a94a6;text-align:center;line-height:1.6}'
        '.method{background:#f0f4f8;border:1px solid #e2e8f0;border-radius:8px;padding:14px 18px;margin-top:18px;font-size:12px;color:#5a6a8c}'
        '</style></head><body><div class="wrap">'
        '<header><p class="kicker">Agent Engine Optimization Report</p><h1>%s</h1>'
        '<div class="scorewrap"><div class="gauge"><div class="inner"><div>'
        '<div class="num">%s</div><div class="lbl">/ %s</div></div></div></div>'
        '<p class="summary">%s</p></div>'
        '<div class="breakdown">%s</div></header>'
        '<section><h2>Findings</h2>%s</section>'
        '<section><h2>Top priority fixes</h2><ol>%s</ol></section>'
        '<div class="method">Scoring methodology: deterministic, threshold-based checks aligned with '
        'the <a href="https://github.com/addyosmani/agentic-seo" target="_blank" rel="noopener noreferrer">agentic-seo</a> framework '
        '(weights: Discovery 25, Parsability 25, Token Efficiency 25, Capability Signaling 15, '
        'Access Control 10). Every point maps to a measurable condition. '
        'Prose analysis generated by Amazon Bedrock (temperature 0) explains scores but does not '
        'influence them. Same page, same score, every time.</div>'
        '<footer>Built with AWS Lambda Durable Functions. '
        'Grounded in: RFC 9309 (Robots Exclusion Protocol), llms.txt proposed standard, '
        'and the AEO framework (Osmani, Google Cloud AI, 2026).</footer>'
        '</div></body></html>'
    ) % (esc(url), gauge, esc(s), esc(url), esc(score), esc(max_score),
         esc(summary), breakdown, rows, fix_items)


# ---------------------------------------------------------------------------
# Steps 4-5: publish + record
# ---------------------------------------------------------------------------

@durable_step
def publish_report(step_context: StepContext, url: str, report: dict) -> dict:
    """Publish HTML and JSON reports to S3."""
    base = f"aeo-reports/{urlparse(url).netloc}{urlparse(url).path.replace('/', '_') or '_root'}"
    json_key = base + ".json"
    html_key = base + ".html"
    s3.put_object(Bucket=ARTIFACTS_BUCKET, Key=json_key,
                  Body=json.dumps(report, indent=2).encode(), ContentType="application/json")
    s3.put_object(Bucket=ARTIFACTS_BUCKET, Key=html_key,
                  Body=render_aeo_html(url, report).encode(), ContentType="text/html")
    step_context.logger.info(f"published {html_key} and {json_key}")
    return {"url": url, "html_key": html_key, "json_key": json_key, "status": "published"}


@durable_step
def record_outcome(step_context: StepContext, execution_name: str, outcome: dict) -> None:
    """Write final result to DynamoDB."""
    ddb.put_item(
        TableName=STATE_TABLE,
        Item={"pk": {"S": execution_name}, "outcome": {"S": json.dumps(outcome)[:38000]}},
    )


# ---------------------------------------------------------------------------
# Durable workflow
# ---------------------------------------------------------------------------

@durable_execution
def handler(event: dict, context: DurableContext) -> dict:
    """
    event = {"execution_name": "aeo-001", "url": "https://docs.example.com/page"}
    """
    execution_name = event.get("execution_name", "aeo-run")
    url = event.get("url", "https://aws.amazon.com/lambda/")

    # Step 1: Fetch page and extract signals
    signals = context.step(fetch_and_parse(url), name="fetch_and_parse")

    # Step 2: Deterministic scoring (no LLM)
    scored = context.step(score_signals(signals), name="score_signals")

    # Step 3: Bedrock prose analysis (temp=0 for reproducibility)
    analysis = context.step(analyze_with_ai(scored), name="analyze_with_ai")

    # Optional: human review gate (disabled by default)
    if APPROVAL_GATE:
        callback = context.create_callback(name="review_report")
        context.logger.info(f"approval gate enabled, callback_id={callback.callback_id}")
        decision = callback.result()
        if isinstance(decision, str):
            try:
                decision = json.loads(decision)
            except (json.JSONDecodeError, TypeError):
                decision = {}
        if not isinstance(decision, dict):
            decision = {}
        if not decision.get("approved"):
            outcome = {
                "execution_name": execution_name, "url": url,
                "status": "rejected", "report": analysis["report"]
            }
            context.step(record_outcome(execution_name, outcome), name="record_outcome")
            return outcome

    # Step 4: Publish to S3
    published = context.step(
        publish_report(url, analysis["report"]),
        name="publish_report",
        config=StepConfig(step_semantics="AT_MOST_ONCE_PER_RETRY"),
    )

    # Step 5: Record to DynamoDB
    outcome = {
        "execution_name": execution_name, "url": url,
        "published": published, "report": analysis["report"]
    }
    context.step(record_outcome(execution_name, outcome), name="record_outcome")
    return outcome
