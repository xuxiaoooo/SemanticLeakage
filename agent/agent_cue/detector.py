"""
Cue Detector: LLM-based depression cue detection with review.
先用LLM检测与抑郁量表相关的词，再用审核LLM过滤误检。
"""

import logging
import re
from typing import Dict, List, Any, Set

from agent.schemas import WordToken
from agent.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ============================================================================
# 第一阶段：检测 Prompt
# ============================================================================
SYSTEM_PROMPT = "You are a clinical psychologist screening for depression indicators."

USER_PROMPT = """Select words that are meaningful clinical indicators of depression.

Only select content words (nouns, adjectives, adverbs) that carry semantic meaning related to:
- Depressed mood or negative emotions
- Loss of interest or pleasure  
- Sleep disturbance
- Fatigue or energy loss
- Appetite or weight changes
- Feelings of worthlessness or guilt
- Concentration difficulties
- Thoughts of death or self-harm

Do NOT select: function words (pronouns, articles, prepositions, conjunctions, nouns), common verbs, or neutral/positive words.

Words: {words}

Return only clinically relevant words, comma-separated. Return NONE if no relevant words found."""


# ============================================================================
# 第二阶段：审核 Prompt（严格过滤）
# ============================================================================
REVIEWER_SYSTEM_PROMPT = "You are a strict clinical linguistics expert reviewing depression-related word detection results."

REVIEWER_USER_PROMPT = """Review the following candidate depression-indicator words and REMOVE any that are NOT valid.

KEEP ONLY words that meet ALL criteria:
1. Directly express negative emotional states (e.g., sad, hopeless, anxious, depressed, lonely)
2. Describe depression symptoms explicitly (e.g., insomnia, fatigue, suicidal, worthless)
3. Are specific clinical/emotional terms, NOT general vocabulary

MUST REMOVE these categories (even if they seem related):
- Function words: pronouns (my, I), articles (the, a), prepositions (of, in, to), conjunctions (and, but, if)
- Common verbs: feel, felt, get, got, been, want, going, thought, wish, care, have, had, do, did
- Neutral nouns: things, people, time, day, life, way, work, place
- Vague adjectives: little, some, ever, much, many, other
- Filler words: like, just, really, actually, basically
- Body parts unless in clinical context: head, heart, body
- Time words: always, never, sometimes, often

Candidate words: {words}

Output ONLY the words that pass ALL criteria, comma-separated.
Output NONE if no words qualify.
Be extremely strict - when in doubt, REMOVE the word."""


class CueReviewer:
    """审核检测结果，过滤误检词"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def review(self, candidate_words: Set[str]) -> Set[str]:
        """
        审核候选词，返回通过审核的词。

        Args:
            candidate_words: 第一阶段检测出的候选词集合

        Returns:
            通过审核的词集合
        """
        if not candidate_words:
            return set()

        # 调用LLM审核
        user_message = REVIEWER_USER_PROMPT.format(words=", ".join(sorted(candidate_words)))

        try:
            response = self.llm_client.client.chat.completions.create(
                model=self.llm_client.model,
                messages=[
                    {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0,
                max_tokens=300,
            )
            result = response.choices[0].message.content.strip()
            logger.info("Reviewer response: %s", result)
        except Exception as e:
            logger.error("LLM review failed: %s", e)
            # 审核失败时返回原始结果
            return candidate_words

        # 解析审核结果
        if result.upper() == "NONE" or not result:
            logger.info("Reviewer rejected all candidates")
            return set()

        # 提取通过审核的词
        approved_words = set()
        for word in result.split(","):
            word = word.strip().lower()
            word = re.sub(r"[^\w\s-]", "", word)
            if word and word in candidate_words:
                approved_words.add(word)

        logger.info("Reviewer approved %d/%d words: %s",
                   len(approved_words), len(candidate_words), approved_words)
        
        # 记录被过滤的词
        rejected = candidate_words - approved_words
        if rejected:
            logger.info("Reviewer rejected: %s", rejected)

        return approved_words


class CueDetector:
    """Detect depression-related cues via LLM with review."""

    def __init__(self, llm_client: LLMClient, enable_review: bool = True):
        """
        Initialize detector.

        Args:
            llm_client: LLM client for detection
            enable_review: 是否启用二次审核（默认启用）
        """
        self.llm_client = llm_client
        self.enable_review = enable_review
        self.reviewer = CueReviewer(llm_client) if enable_review else None
        logger.info("CueDetector initialized (review=%s)", enable_review)

    def _parse_llm_words(self, result: str) -> Set[str]:
        """解析LLM返回的词列表"""
        words = set()
        for word in result.split(","):
            word = word.strip().lower()
            word = re.sub(r"[^\w\s-]", "", word)
            if word:
                words.add(word)
        return words

    def _detect_batch(self, unique_words: List[str]) -> Set[str]:
        """对一批词进行检测"""
        user_message = USER_PROMPT.format(words=", ".join(unique_words))

        try:
            response = self.llm_client.client.chat.completions.create(
                model=self.llm_client.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0,
                max_tokens=300,
            )
            result = response.choices[0].message.content.strip()
            logger.info("Detection response: %s", result)
        except Exception as e:
            logger.error("LLM detection failed: %s", e)
            return set()

        if result.upper() == "NONE" or not result:
            return set()

        return self._parse_llm_words(result)

    def detect(self, words: List[WordToken]) -> List[Dict[str, Any]]:
        """
        Detect depression-related cues from word list using LLM.
        对于长文本，自动分批处理以避免 API 限制。

        Args:
            words: List of WordToken from MultiScaleTranscript

        Returns:
            List of detected cue items with span info
        """
        if not words:
            return []

        # 构建word文本列表（去重，保留唯一词）
        unique_words = list(set(w.text.lower().strip() for w in words if w.text.strip()))
        if not unique_words:
            return []

        # ========== 第一阶段：检测（支持分批处理长文本）==========
        # 每批最多处理 500 个词，避免超出 API token 限制
        batch_size = 500
        detected_words = set()

        if len(unique_words) <= batch_size:
            # 短文本，一次处理
            detected_words = self._detect_batch(unique_words)
        else:
            # 长文本，分批处理
            logger.info("Long text detected (%d unique words), processing in batches", len(unique_words))
            for i in range(0, len(unique_words), batch_size):
                batch = unique_words[i:i + batch_size]
                logger.info("Processing batch %d/%d (%d words)",
                           i // batch_size + 1,
                           (len(unique_words) + batch_size - 1) // batch_size,
                           len(batch))
                batch_detected = self._detect_batch(batch)
                detected_words.update(batch_detected)

        if not detected_words:
            logger.info("No depression-related cues detected")
            return []

        logger.info("Stage 1 detected %d terms: %s", len(detected_words), detected_words)

        # ========== 第二阶段：审核 ==========
        if self.enable_review and self.reviewer and detected_words:
            detected_words = self.reviewer.review(detected_words)
            if not detected_words:
                logger.info("All candidates rejected by reviewer")
                return []

        logger.info("Final approved %d terms: %s", len(detected_words), detected_words)

        # 根据检测到的词，从原始words中找对应的span
        cues = []
        for word_token in words:
            normalized = word_token.text.lower().strip()
            if normalized in detected_words:
                cues.append({
                    "word_id": word_token.id,
                    "text": word_token.text,
                    "start": word_token.start,
                    "end": word_token.end,
                    "speaker": word_token.speaker,
                })

        logger.info("Detection complete: %d cue instances found", len(cues))
        return cues

    def detect_from_sentences(self, sentences: List) -> List[Dict[str, Any]]:
        """
        ManDIC segment 模式：从 sentences (整句) 中检测抑郁相关词。

        Args:
            sentences: List of SentenceSpan from MultiScaleTranscript

        Returns:
            List of detected cue items with sentence_id and span info
        """
        if not sentences:
            return []

        # 从所有句子中提取唯一词（简单按空格和标点分割）
        import re
        all_words = []
        for sent in sentences:
            text = sent.text or ""
            # 中文按字符，英文按空格
            words = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', text.lower())
            all_words.extend(words)

        unique_words = list(set(all_words))
        if not unique_words:
            return []

        logger.info("Segment mode: extracted %d unique words from %d sentences",
                   len(unique_words), len(sentences))

        # ========== 第一阶段：检测 ==========
        batch_size = 500
        detected_words = set()

        if len(unique_words) <= batch_size:
            detected_words = self._detect_batch(unique_words)
        else:
            logger.info("Long text detected (%d unique words), processing in batches", len(unique_words))
            for i in range(0, len(unique_words), batch_size):
                batch = unique_words[i:i + batch_size]
                batch_detected = self._detect_batch(batch)
                detected_words.update(batch_detected)

        if not detected_words:
            logger.info("No depression-related cues detected")
            return []

        logger.info("Stage 1 detected %d terms: %s", len(detected_words), detected_words)

        # ========== 第二阶段：审核 ==========
        if self.enable_review and self.reviewer and detected_words:
            detected_words = self.reviewer.review(detected_words)
            if not detected_words:
                logger.info("All candidates rejected by reviewer")
                return []

        logger.info("Final approved %d terms: %s", len(detected_words), detected_words)

        # 找到包含检测词的句子
        cues = []
        for sent in sentences:
            text_lower = (sent.text or "").lower()
            # 检查句子是否包含任何检测词
            matching_words = [w for w in detected_words if w in text_lower]
            if matching_words:
                cues.append({
                    "sentence_id": sent.id,
                    "text": ", ".join(matching_words),  # 该句中检测到的词
                    "start": sent.start,
                    "end": sent.end,
                    "speaker": sent.speaker,
                    "sentence_text": sent.text,  # 完整句子文本（可选）
                })

        logger.info("Segment mode detection complete: %d sentences with cues", len(cues))
        return cues


__all__ = ["CueDetector", "CueReviewer"]
