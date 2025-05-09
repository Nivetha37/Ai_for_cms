from groq import Groq

# Initialize Groq client
client = Groq()


def llama_guard_check(text):
    messages = [
        {
            "role": "system",
            "content": "You are a content moderation assistant. Check the following LinkedIn post for safety and appropriateness.",
        },
        {"role": "user", "content": text},
    ]

    completion = client.chat.completions.create(
        model="llama-guard-3-8b",
        messages=messages,
        temperature=0.0,
        max_tokens=1024,
        top_p=1,
        stream=False,
    )

    return completion.choices[0].message.content


def guardrails_check(post_text):
    response = llama_guard_check(post_text)

    # Define heuristic for failure
    failure_indicators = [
        "unsafe",
        "harmful",
        "toxic",
        "flagged",
        "not appropriate",
        "violates",
        "sensitive",
        "offensive",
    ]

    if any(keyword in response.lower() for keyword in failure_indicators):
        return False, f"Flagged by LLaMA Guard: {response}"

    return True, ""


def evaluate_post(post_text, expected_tag, expected_length):
    issues = []
    score = 1.0

    lines = post_text.strip().split("\n")
    line_count = len(lines)

    if expected_length == "Short" and line_count > 5:
        issues.append("Too long for 'Short'")
        score -= 0.2
    elif expected_length == "Medium" and not (5 <= line_count <= 10):
        issues.append("Line count mismatch for 'Medium'")
        score -= 0.2
    elif expected_length == "Long" and line_count < 10:
        issues.append("Too short for 'Long'")
        score -= 0.2

    # Simple relevance check
    if expected_tag.lower() not in post_text.lower():
        issues.append("Topic tag not mentioned explicitly")
        score -= 0.2

    return {"score": round(score, 2), "issues": issues}


def safe_generate_post(length, language, tag, generator_fn):
    post = generator_fn(length, language, tag)

    # Guardrails first
    passed, reason = guardrails_check(post)
    if not passed:
        raise ValueError(f"🚨 Guardrail failed: {reason}")

    # Evaluation
    eval_result = evaluate_post(post, tag, length)
    if eval_result["score"] < 0.7:
        raise ValueError(f"⚠️ Post quality too low: {eval_result['issues']}")

    return post, eval_result
