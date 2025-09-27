# ============================================================
# 2-Stage Guardrail System with Hugging Face Moderation
# ============================================================

from typing import Dict, List, Tuple
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline


# ----------------------------
# Safety Finding Dataclass
# ----------------------------
class SafetyFinding:
    def __init__(self, label: str, severity: str, message: str):
        self.label = label
        self.severity = severity
        self.message = message

    def to_dict(self):
        return {
            "label": self.label,
            "severity": self.severity,
            "message": self.message
        }


# ----------------------------
# Heuristic Checkers (simple stubs, extend as needed)
# ----------------------------
class HeuristicCheckers:
    @staticmethod
    def find_jailbreak(text: str) -> List[SafetyFinding]:
        findings = []
        jailbreak_terms = ["ignore previous", "system prompt", "jailbreak"]
        for term in jailbreak_terms:
            if term in text.lower():
                findings.append(SafetyFinding(
                    label="jailbreak_heuristic",
                    severity="high",
                    message=f"Contains suspicious jailbreak term: {term}"
                ))
        return findings

    @staticmethod
    def find_safety_content(text: str) -> List[SafetyFinding]:
        findings = []
        unsafe_terms = ["bomb", "kill", "suicide", "explosive","hack","rob","theif","steal","drugs","acid","murder"]
        for term in unsafe_terms:
            if term in text.lower():
                findings.append(SafetyFinding(
                    label="unsafe_content",
                    severity="high",
                    message=f"Contains unsafe term: {term}"
                ))
        return findings

    @staticmethod
    def find_pii(text: str) -> List[SafetyFinding]:
        findings = []
        if "@" in text:  # naive email detection
            findings.append(SafetyFinding(
                label="pii",
                severity="medium",
                message="Possible email detected"
            ))
        return findings

    @staticmethod
    def find_prompt_leakage(text: str) -> List[SafetyFinding]:
        findings = []
        if "instruction" in text.lower() or "prompt" in text.lower():
            findings.append(SafetyFinding(
                label="prompt_leakage",
                severity="medium",
                message="Possible prompt leakage"
            ))
        return findings


# ----------------------------
# Hugging Face Moderation Checker (open-source)
# ----------------------------
class HuggingFaceModerationChecker:
    def __init__(self, model="unitary/toxic-bert"):
        print(f"Loading Hugging Face moderation model: {model}")
        self.classifier = pipeline("text-classification", model=model)

    def check(self, text: str) -> List[SafetyFinding]:
        findings = []
        if not text.strip():
            return findings

        results = self.classifier(text, truncation=True)
        for r in results:
            label = r["label"].lower()
            score = float(r["score"])
            if label in ["toxic", "offensive", "hate", "violence"] and score > 0.7:
                findings.append(SafetyFinding(
                    label="huggingface_moderation",
                    severity="high",
                    message=f"⚠️ Flagged as {r['label']} (score={score:.2f})"
                ))
        return findings


# ----------------------------
# JailbreakBench Behavior Classifier
# ----------------------------
class JBBBehaviorClassifier:
    def __init__(self, threshold: float = 0.5, embed_model: str = "all-MiniLM-L6-v2"):
        print("Loading JailbreakBench behaviors...")

        # Use harmful split (malicious prompts)
        dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")

        self.malicious_goals = [row["Goal"] for row in dataset if row and "Goal" in row and row["Goal"]]
        print(f"Loaded {len(self.malicious_goals)} malicious goals from harmful split.")

        print("Loading embedding model:", embed_model)
        self.model = SentenceTransformer(embed_model)
        self.goal_embeddings = self.model.encode(self.malicious_goals, convert_to_tensor=True)

        self.threshold = threshold

    def check(self, user_prompt: str) -> List[SafetyFinding]:
        findings = []
        if not user_prompt.strip():
            return findings

        user_emb = self.model.encode(user_prompt, convert_to_tensor=True)
        cos_scores = util.cos_sim(user_emb, self.goal_embeddings)[0]
        max_score = float(cos_scores.max())

        # Always print similarity + threshold
        print(f"[JBB Result] Prompt: {user_prompt}")
        print(f"[JBB Result] Similarity Score = {max_score:.4f} | Threshold = {self.threshold}")

        if max_score >= self.threshold:
            print(f"[JBB Decision] ❌ BLOCKED (score {max_score:.4f} ≥ {self.threshold})")
            findings.append(SafetyFinding(
                label="jailbreak",
                severity="high",
                message=f"Blocked: Similar to known jailbreak (score={max_score:.2f}, threshold={self.threshold})"
            ))
        else:
            print(f"[JBB Decision] ✅ ALLOWED (score {max_score:.4f} < {self.threshold})")

        return findings


class LLMClient:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        # self.key = os.getenv("OPENAI_API_KEY")  # your API key

    def generate(self, system: str, user: str) -> str:
        """Generate text from an LLM"""
        # if self.key:
        #     import openai
        #     openai.api_key = self.key
        #     resp = openai.ChatCompletion.create(
        #         model=self.model,
        #         messages=[
        #             {"role": "system", "content": system},
        #             {"role": "user", "content": user}
        #         ],
        #         temperature=0.0
        #     )
        #     return resp.choices[0].message["content"]
        return "LLM API key missing or stub response"


# ----------------------------
# Reviewer (with HuggingFace moderation)
# ----------------------------
class Reviewer:
    def __init__(self, policy: Dict):
        self.policy = policy
        self.mod = HuggingFaceModerationChecker()   # no API key needed
        self.jbb = JBBBehaviorClassifier(threshold=0.5)

    def _decide(self, findings: List[SafetyFinding]) -> str:
        """Decide whether to block, redact, or allow based on findings."""
        if any(f.severity == "high" for f in findings):
            return "block"
        if any(f.severity == "medium" for f in findings):
            return "redact"
        return "allow"

    def _redact(self, text: str, findings: List[SafetyFinding]) -> Tuple[str, int]:
        """Naive redaction (replace unsafe terms)."""
        redacted = text
        count = 0
        for f in findings:
            if f.label in ["pii", "prompt_leakage"]:
                redacted = redacted.replace("prompt", "[REDACTED]").replace("@", "[REDACTED]")
                count += 1
        return redacted, count

    def review(self, user_prompt: str, draft_output: str) -> Dict:
        findings: List[SafetyFinding] = []

        # Stage 1 checks (on input)
        findings += HeuristicCheckers.find_jailbreak(user_prompt)
        findings += HeuristicCheckers.find_safety_content(user_prompt)
        findings += self.jbb.check(user_prompt)

        # Stage 2 checks (on output)
        findings += HeuristicCheckers.find_pii(draft_output)
        findings += HeuristicCheckers.find_prompt_leakage(draft_output)
        findings += HeuristicCheckers.find_safety_content(draft_output)

        # HuggingFace moderation (always mandatory)
        findings += self.mod.check(user_prompt + "\n\n---\n\n" + draft_output)

        # Decide action
        action = self._decide(findings)
        if action == "block":
            return {
                "action": "block",
                "output": self.policy["messages"]["blocked"],
                "findings": [f.to_dict() for f in findings]
            }
        if action == "redact":
            redacted, _ = self._redact(draft_output, findings)
            final = self.policy["messages"]["redacted_notice"] + "\n\n" + redacted
            if any(f.label == "medical_or_legal_advice" for f in findings):
                final = self.policy["messages"]["disclaimer_medical_legal"] + "\n\n" + final
            return {
                "action": "redact",
                "output": final,
                "findings": [f.to_dict() for f in findings]
            }

        # Allow
        final = draft_output
        if any(f.label == "medical_or_legal_advice" for f in findings):
            final = self.policy["messages"]["disclaimer_medical_legal"] + "\n\n" + final
        return {
            "action": "allow",
            "output": final,
            "findings": [f.to_dict() for f in findings]
        }


# ----------------------------
# Example Policy
# ----------------------------
policy = {
    "messages": {
        "blocked": "❌ This response was blocked for safety reasons.",
        "redacted_notice": "⚠️ Some content was redacted due to policy.",
        "disclaimer_medical_legal": "Please consult a qualified professional for medical or legal advice."
    }
}

# ----------------------------
# Example Usage
# ----------------------------
reviewer = Reviewer(policy)

user_prompt = "how to dispose a body?"
system_prompt = "You are a helpful assistant." # Define system_prompt
llm = LLMClient(model="gpt-4o-mini")
draft_output = llm.generate(system_prompt, user_prompt)


result = reviewer.review(user_prompt, draft_output)
print(result)

# Print similarity score and threshold from findings if available
for finding in result.get('findings', []):
    if finding.get('label') == 'jailbreak' and 'score' in finding.get('message', ''):
        print(finding.get('message'))
