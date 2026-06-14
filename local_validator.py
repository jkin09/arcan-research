# ============================================================================
# ARCAN-X — representative excerpt (adversarial validator)
#
# This is one real, self-contained component from ARCAN-X, shared as a sample
# of the engineering approach. It uses a separate model (different training
# distribution from the primary) as an adversarial critic: it finds the worst
# flaw in a research output first, then scores. The final job grade averages
# the primary model's self-score with this independent score, and persistent
# divergence is tracked as an overconfidence signal.
#
# The full system is private. This file runs standalone against a local Ollama.
# ============================================================================

"""
ARCAN-X — backend/learning/local_validator.py
Independent adversarial validator using deepseek-r1:14b.

Primary model:  qwen2.5:14b   (research, reasoning, chat)
Validator:      deepseek-r1:14b (adversarial scorer — different training distribution)

deepseek-r1 uses chain-of-thought reasoning. We exploit this:
- Prompt it to find the WORST flaw first, then score.
- Extract <think> block as the critique.
- Score = 100 minus severity of worst flaw found.

This means the validator is genuinely adversarial, not just a second opinion.
Final job grade = average(qwen_score, deepseek_score).

Tracks divergence over time: persistent qwen > deepseek delta = qwen overconfidence signal.
"""
from __future__ import annotations
import asyncio, json, logging, re, time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("arcan.local_validator")

VALIDATION_LOG = Path("data/selfmod/local_validations.jsonl")
VALIDATION_LOG.parent.mkdir(parents=True, exist_ok=True)

VALIDATOR_MODEL   = "deepseek-r1:14b"   # adversarial scorer
VALIDATOR_TIMEOUT = 240                  # 14b needs more time than 3b

ADVERSARIAL_SYSTEM = """You are an adversarial research evaluator. Your job is NOT to validate — it is to find flaws.

For any research output presented to you:
1. Find the single most serious error, gap, or fabrication present.
2. Check specifically: invented citations, wrong units, implausible numbers, incomplete sections, circular reasoning.
3. After finding the worst flaw, assign a score 0-100 where:
   - 90-100: No serious errors. Findings are specific, numerical, and well-supported.
   - 70-89: Minor gaps only. Core claims are defensible.
   - 50-69: Significant gap or one major error present.
   - 0-49: Fabrications, unit errors, or fundamental reasoning failures.

Respond in EXACTLY this format — nothing else:
FLAW: <one sentence describing the worst problem found, or "None" if output is solid>
Score: <integer 0-100>"""


async def validate_with_llama(
    job_id: str,
    title: str,
    result: str,
    primary_score: int,
    ollama_host: str = "http://localhost:11434",
) -> Optional[dict]:
    """
    Score a job result using deepseek-r1:14b as adversarial critic.
    Returns averaged score and validation record.
    Interface-compatible with the original llama3.2 validator.
    """
    if not result or len(result.strip()) < 50:
        return None

    result_truncated = result[:3000]

    messages = [
        {"role": "system", "content": ADVERSARIAL_SYSTEM},
        {"role": "user",   "content": f"Task: {title}\n\nOutput to evaluate:\n{result_truncated}"},
    ]

    try:
        import httpx

        async with httpx.AsyncClient(timeout=VALIDATOR_TIMEOUT) as client:
            resp = await client.post(
                f"{ollama_host}/api/chat",
                json={
                    "model":    VALIDATOR_MODEL,
                    "messages": messages,
                    "stream":   False,
                    "options":  {"temperature": 0.1, "num_predict": 300},
                },
            )
            if resp.status_code != 200:
                logger.debug(f"LocalValidator: ollama returned {resp.status_code}")
                return None
            raw_text = resp.json().get("message", {}).get("content", "").strip()

        # Strip deepseek-r1 <think> blocks — extract critique from them
        think_text = ""
        think_match = re.search(r'<think>(.*?)</think>', raw_text, re.DOTALL)
        if think_match:
            think_text = think_match.group(1).strip()[:400]
            raw_text   = raw_text[think_match.end():].strip()

        logger.info(f"LocalValidator deepseek raw: {repr(raw_text[:200])}")

        # Parse FLAW line
        flaw = ""
        flaw_match = re.search(r'FLAW:\s*(.+)', raw_text, re.IGNORECASE)
        if flaw_match:
            flaw = flaw_match.group(1).strip()[:200]
            if flaw.lower() in ('none', 'none.', 'n/a', 'no flaw', 'no serious flaw'):
                flaw = ""

        # Parse score
        score_match = re.search(
            r'Score[:\s]+(\d{1,3})',
            raw_text, re.IGNORECASE | re.MULTILINE)
        if not score_match:
            # No "Score:" line found. Do NOT grab a random bare number —
            # deepseek's reasoning text is full of stray numbers (frequencies,
            # hypothesis indices) that would corrupt the grade. Fail closed.
            logger.debug(f"LocalValidator: no Score: line parsed, skipping: {raw_text[:150]}")
            return None

        deepseek_score = max(0, min(100, int(score_match.group(1))))
        averaged_score = round((primary_score + deepseek_score) / 2)
        delta          = deepseek_score - primary_score

        # Divergence warning — deepseek consistently lower = qwen overconfidence
        if delta <= -15:
            logger.warning(
                f"ValidatorDivergence {job_id[:8]}: "
                f"qwen={primary_score} deepseek={deepseek_score} "
                f"delta={delta:+d} | FLAW: {flaw[:80]}"
            )

        # Build critique from flaw + think chain
        critique = flaw
        if think_text and not flaw:
            critique = think_text[:200]

        record = {
            "job_id":         job_id,
            "title":          title[:80],
            "primary_score":  primary_score,
            "llama_score":    deepseek_score,   # kept as llama_score for interface compat
            "deepseek_score": deepseek_score,
            "hermes_score":   None,
            "averaged_score": averaged_score,
            "delta":          delta,
            "flaw":           flaw,
            "critique":       critique,
            "reason":         critique,          # interface compat
            "ts":             time.time(),
        }

        try:
            with open(VALIDATION_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

        level = "info" if abs(delta) > 10 else "debug"
        getattr(logger, level)(
            f"LocalValidator {job_id[:8]}: "
            f"qwen={primary_score} deepseek={deepseek_score} "
            f"avg={averaged_score} delta={delta:+d}"
            + (f" | FLAW: {flaw[:60]}" if flaw else "")
        )

        return record

    except asyncio.TimeoutError:
        logger.debug(f"LocalValidator: timeout on {job_id[:8]}")
        return None
    except Exception as e:
        logger.debug(f"LocalValidator error: {e}")
        return None


async def check_model_available(ollama_host: str = "http://localhost:11434") -> bool:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ollama_host}/api/tags")
            if resp.status_code == 200:
                names = [m.get("name", "") for m in resp.json().get("models", [])]
                return any(VALIDATOR_MODEL in n for n in names)
    except Exception:
        pass
    return False


def get_calibration_stats() -> dict:
    """Divergence stats — persistent qwen > deepseek gap = overconfidence signal."""
    if not VALIDATION_LOG.exists():
        return {"count": 0, "model": VALIDATOR_MODEL, "avg_delta": 0, "available": False}

    records = []
    for line in VALIDATION_LOG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    if not records:
        return {"count": 0, "model": VALIDATOR_MODEL, "avg_delta": 0, "available": False}

    deltas  = [r["delta"] for r in records]
    avg_del = sum(deltas) / len(deltas)
    agree   = sum(1 for d in deltas if abs(d) <= 10) / len(deltas)
    flaws   = [r.get("flaw", "") for r in records if r.get("flaw")]

    # Overconfidence detection: qwen scores consistently higher than deepseek
    overconfident = avg_del < -8

    return {
        "count":            len(records),
        "model":            VALIDATOR_MODEL,
        "avg_delta":        round(avg_del, 1),
        "agreement_pct":    round(agree * 100, 1),
        "overconfidence":   overconfident,
        "available":        True,
        "recent_deltas":    deltas[-5:],
        "common_flaws":     flaws[-5:],
        "message": (
            f"{'⚠ qwen overconfidence detected: ' if overconfident else ''}"
            f"deepseek agrees within 10pts {agree*100:.0f}% of the time. "
            f"Avg divergence: {abs(avg_del):.1f}pts "
            f"({'qwen higher' if avg_del < 0 else 'deepseek higher'})."
        ) if len(records) >= 5 else "Need 5+ validations for calibration stats",
    }
