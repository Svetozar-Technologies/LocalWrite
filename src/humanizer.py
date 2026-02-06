"""
Advanced AI Humanizer - Designed to bypass AI detection tools like Copyleaks.

Key insight: AI text is too "perfect". We need to deliberately imperfect it
with patterns that match human writing quirks.

Detection signals we target:
1. Perplexity - AI uses predictable words, humans use unexpected choices
2. Burstiness - AI is uniform, humans vary sentence length dramatically
3. Structure - AI is grammatically perfect, humans use fragments/run-ons
4. Flow - AI is smooth, humans have asides/tangents/interruptions
"""

import re
import random
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


@dataclass
class HumanizerConfig:
    """Configuration for the humanizer."""
    model_path: str = ""
    n_ctx: int = 4096
    n_threads: int = 4
    n_gpu_layers: int = 0
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 60
    repeat_penalty: float = 1.2
    max_tokens: int = 512


# Simpler, more direct prompts for Qwen
REWRITE_PROMPTS = [
    """Rewrite this in your own words. Be natural and casual. Use contractions.
Just output the rewritten text, nothing else.""",

    """Say this differently, like you're explaining to a friend. Keep it simple.
Only output the new version.""",

    """Rephrase this casually. Use everyday language. Be conversational.
Output only the result.""",
]


# Word replacements: common AI words -> more human alternatives
WORD_SUBSTITUTIONS = {
    # Formal -> Casual
    "utilize": ["use", "work with", "go with"],
    "utilizes": ["uses", "works with"],
    "utilizing": ["using", "working with"],
    "obtain": ["get", "grab", "pick up"],
    "obtain": ["gets", "grabs"],
    "sufficient": ["enough", "plenty of"],
    "insufficient": ["not enough", "too little"],
    "commence": ["start", "begin", "kick off"],
    "terminate": ["end", "stop", "finish"],
    "facilitate": ["help", "make easier", "assist with"],
    "implement": ["set up", "put in place", "roll out"],
    "subsequently": ["then", "after that", "later"],
    "previously": ["before", "earlier", "back then"],
    "additionally": ["also", "plus", "on top of that"],
    "furthermore": ["also", "and", "plus"],
    "moreover": ["also", "and", "what's more"],
    "however": ["but", "though", "still"],
    "therefore": ["so", "that's why", "which means"],
    "consequently": ["so", "as a result", "because of this"],
    "nevertheless": ["still", "even so", "but still"],
    "numerous": ["many", "lots of", "a bunch of"],
    "significant": ["big", "major", "important"],
    "approximately": ["about", "around", "roughly"],
    "immediately": ["right away", "instantly", "straight away"],
    "frequently": ["often", "a lot", "regularly"],
    "occasionally": ["sometimes", "now and then", "once in a while"],
    "demonstrate": ["show", "prove", "make clear"],
    "indicates": ["shows", "means", "suggests"],
    "regarding": ["about", "on", "when it comes to"],
    "concerning": ["about", "around", "regarding"],
    "possess": ["have", "own", "got"],
    "acquire": ["get", "pick up", "gain"],
    "provide": ["give", "offer", "share"],
    "require": ["need", "must have", "call for"],
    "assist": ["help", "aid", "support"],
    "attempt": ["try", "go for", "take a shot at"],
    "determine": ["figure out", "find out", "decide"],
    "establish": ["set up", "create", "build"],
    "maintain": ["keep", "hold onto", "stick with"],
    "observe": ["see", "notice", "watch"],
    "perform": ["do", "carry out", "handle"],
    "purchase": ["buy", "get", "pick up"],
    "receive": ["get", "be given"],
    "remain": ["stay", "keep being", "stick around"],
    "request": ["ask for", "want"],
    "select": ["pick", "choose", "go with"],
    "substantial": ["big", "large", "major"],
    "sufficient": ["enough", "plenty"],
    "currently": ["now", "right now", "at the moment"],
    "primarily": ["mainly", "mostly", "for the most part"],
    "essentially": ["basically", "really", "pretty much"],
    "specifically": ["especially", "particularly", "in particular"],
    "extremely": ["really", "super", "very"],
    "highly": ["really", "very", "super"],
    "rapidly": ["quickly", "fast"],
    "efficiently": ["well", "smoothly"],
    "effectively": ["well", "properly"],
    "individuals": ["people", "folks"],
    "individual": ["person", "someone"],
    "children": ["kids"],
    "purchase": ["buy", "get"],
    "residence": ["home", "place"],
    "vehicle": ["car"],
    "beverage": ["drink"],
    "cuisine": ["food"],
    "garments": ["clothes"],
    "physician": ["doctor"],
    "occupation": ["job"],
    "compensation": ["pay", "salary"],
    "initiate": ["start", "begin"],
    "finalize": ["finish", "wrap up"],
    "optimize": ["improve", "make better"],
    "leverage": ["use", "take advantage of"],
    "streamline": ["simplify", "make easier"],
    "prioritize": ["focus on", "put first"],
}

# Phrases that scream "AI wrote this"
AI_PHRASES_TO_REMOVE = [
    "it's important to note that",
    "it is important to note that",
    "it's worth noting that",
    "it is worth noting that",
    "it should be noted that",
    "in today's world",
    "in today's society",
    "in this day and age",
    "at the end of the day",
    "when it comes to",
    "in terms of",
    "on the other hand",
    "in order to",
    "due to the fact that",
    "for the purpose of",
    "in the event that",
    "at this point in time",
    "in the near future",
    "a wide range of",
    "a variety of",
    "a number of",
    "the fact that",
    "it goes without saying",
    "needless to say",
    "as a matter of fact",
    "in conclusion",
    "to summarize",
    "in summary",
    "all things considered",
    "taking everything into account",
    "plays a crucial role",
    "plays an important role",
    "serves as a",
    "acts as a",
    "functions as a",
    # More AI patterns
    "dive into",
    "dive in",
    "delve into",
    "delve in",
    "it's crucial",
    "it is crucial",
    "it's essential",
    "it is essential",
    "first and foremost",
    "last but not least",
    "in light of",
    "with that being said",
    "that being said",
    "having said that",
    "as previously mentioned",
    "as mentioned earlier",
    "as stated above",
    "in other words",
    "to put it simply",
    "simply put",
    "broadly speaking",
    "generally speaking",
    "interestingly enough",
    "importantly",
    "significantly",
    "remarkably",
    "notably",
    "crucially",
    "essentially",
    "fundamentally",
    "ultimately",
    "specifically",
]

# Human filler phrases to inject
HUMAN_FILLERS = [
    "honestly",
    "actually",
    "basically",
    "really",
    "pretty much",
    "kind of",
    "sort of",
    "you know",
    "I mean",
    "like",
    "anyway",
    "so yeah",
    "right",
    "obviously",
    "clearly",
    "definitely",
]

# Parenthetical asides humans use
PARENTHETICAL_ASIDES = [
    "(which makes sense)",
    "(honestly)",
    "(obviously)",
    "(no surprise there)",
    "(not gonna lie)",
    "(to be fair)",
    "(in my experience)",
    "(believe it or not)",
    "(interestingly enough)",
    "(fun fact)",
    "(spoiler alert)",
    "(side note)",
    "(quick note)",
]

# Sentence starters that feel human
HUMAN_STARTERS = [
    "Look, ",
    "See, ",
    "Thing is, ",
    "Here's the deal: ",
    "Real talk: ",
    "Honestly, ",
    "Truth is, ",
    "Point is, ",
    "Bottom line: ",
    "So basically, ",
    "The way I see it, ",
    "Let's be real, ",
    "Not gonna lie, ",
]

# Transition words/phrases
CASUAL_TRANSITIONS = [
    "And ",
    "But ",
    "So ",
    "Plus ",
    "Also ",
    "Now ",
    "Then ",
    "Anyway, ",
    "Besides, ",
    "Still, ",
    "Though ",
    "Yet ",
]


class Humanizer:
    """
    Advanced humanizer that bypasses AI detection through statistical manipulation.
    """

    def __init__(self, config: Optional[HumanizerConfig] = None):
        self.config = config or HumanizerConfig()
        self.model: Optional[Llama] = None
        self.is_loaded = False

    def load_model(
        self,
        model_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """Load the LLM model."""
        if Llama is None:
            if progress_callback:
                progress_callback("Error: llama-cpp-python not installed")
            return False

        import os
        path = model_path or self.config.model_path

        if not path or not os.path.exists(path):
            if progress_callback:
                progress_callback(f"Error: Model not found: {path}")
            return False

        try:
            if progress_callback:
                progress_callback("Loading model...")

            n_gpu = self.config.n_gpu_layers
            if n_gpu == 0:
                import platform
                if platform.system() == "Darwin":
                    n_gpu = -1

            self.model = Llama(
                model_path=path,
                n_ctx=self.config.n_ctx,
                n_threads=self.config.n_threads,
                n_gpu_layers=n_gpu,
                verbose=False
            )

            self.config.model_path = path
            self.is_loaded = True

            if progress_callback:
                progress_callback("Model loaded!")

            return True

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {e}")
            return False

    def unload_model(self):
        """Unload model to free memory."""
        if self.model:
            del self.model
            self.model = None
        self.is_loaded = False

    def humanize(
        self,
        text: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Humanize text to bypass AI detection.
        Multi-stage processing for maximum effectiveness.
        """
        if not self.is_loaded or not self.model:
            return text

        if not text or not text.strip():
            return text

        try:
            # Stage 1: LLM rewrite
            if progress_callback:
                progress_callback("Stage 1: Rewriting with LLM...")

            result = self._llm_rewrite(text)

            # Stage 2: Remove AI-telltale phrases
            if progress_callback:
                progress_callback("Stage 2: Removing AI patterns...")

            result = self._remove_ai_phrases(result)

            # Stage 3: Word substitutions (formal -> casual)
            if progress_callback:
                progress_callback("Stage 3: Casualizing vocabulary...")

            result = self._substitute_words(result)

            # Stage 4: Apply contractions
            if progress_callback:
                progress_callback("Stage 4: Adding contractions...")

            result = self._apply_contractions(result)

            # Stage 5: Inject burstiness (vary sentence lengths)
            if progress_callback:
                progress_callback("Stage 5: Adding sentence variety...")

            result = self._inject_burstiness(result)

            # Stage 6: Add human quirks
            if progress_callback:
                progress_callback("Stage 6: Adding human touches...")

            result = self._add_human_quirks(result)

            # Stage 7: Final cleanup
            if progress_callback:
                progress_callback("Stage 7: Final polish...")

            result = self._final_cleanup(result)

            return result

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {e}")
            return text

    def _llm_rewrite(self, text: str) -> str:
        """Stage 1: Use LLM to rewrite the text."""
        paragraphs = self._split_paragraphs(text)
        rewritten = []

        for para in paragraphs:
            if not para.strip():
                rewritten.append(para)
                continue

            if len(para.split()) < 5:  # Very short, keep as is
                rewritten.append(para)
                continue

            prompt_template = random.choice(REWRITE_PROMPTS)

            # Qwen ChatML format
            prompt = f"""<|im_start|>system
{prompt_template}<|im_end|>
<|im_start|>user
{para}<|im_end|>
<|im_start|>assistant
"""

            response = self.model(
                prompt,
                max_tokens=min(len(para.split()) * 4, self.config.max_tokens),
                temperature=self.config.temperature + random.uniform(-0.1, 0.15),
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repeat_penalty=self.config.repeat_penalty,
                stop=["<|im_end|>", "<|im_start|>", "\n\nReferences:", "\n\nSources:"],
                echo=False
            )

            result = response["choices"][0]["text"].strip()
            result = self._clean_llm_output(result)

            # Validate output
            if result and len(result.split()) >= len(para.split()) * 0.4:
                rewritten.append(result)
            else:
                rewritten.append(para)

        return "\n\n".join(rewritten)

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs]

    def _clean_llm_output(self, text: str) -> str:
        """Clean LLM output artifacts."""
        result = text.strip()

        # Remove common prefixes
        prefixes = [
            "here's", "here is", "rewritten:", "output:", "sure,", "okay,",
            "here you go:", "the rewritten", "paraphrased:", "certainly,"
        ]
        result_lower = result.lower()
        for prefix in prefixes:
            if result_lower.startswith(prefix):
                result = result[len(prefix):].strip()
                result_lower = result.lower()

        # Remove quotes if wrapped
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]

        # Remove fake citations
        result = re.sub(r'\([A-Z][a-z]+\s+et\s+al\.,?\s*\d{4}\)', '', result)
        result = re.sub(r'\[\d+\]', '', result)

        return result.strip()

    def _remove_ai_phrases(self, text: str) -> str:
        """Stage 2: Remove phrases that scream 'AI wrote this'."""
        result = text

        for phrase in AI_PHRASES_TO_REMOVE:
            # Case insensitive replacement
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            result = pattern.sub('', result)

        # Clean up double spaces
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'\s+([.,!?])', r'\1', result)

        return result.strip()

    def _substitute_words(self, text: str) -> str:
        """Stage 3: Replace formal/AI words with casual alternatives."""
        result = text

        for formal, casual_list in WORD_SUBSTITUTIONS.items():
            # Match whole words only, case insensitive
            pattern = re.compile(r'\b' + formal + r'\b', re.IGNORECASE)

            def replace_match(match):
                replacement = random.choice(casual_list)
                # Preserve capitalization
                if match.group()[0].isupper():
                    return replacement.capitalize()
                return replacement

            # Only replace sometimes (70% chance) to maintain variety
            if random.random() < 0.7:
                result = pattern.sub(replace_match, result)

        return result

    def _apply_contractions(self, text: str) -> str:
        """Stage 4: Convert to contractions."""
        contractions = [
            (r"\bdo not\b", "don't"),
            (r"\bDo not\b", "Don't"),
            (r"\bcannot\b", "can't"),
            (r"\bCannot\b", "Can't"),
            (r"\bwill not\b", "won't"),
            (r"\bWill not\b", "Won't"),
            (r"\bwould not\b", "wouldn't"),
            (r"\bWould not\b", "Wouldn't"),
            (r"\bcould not\b", "couldn't"),
            (r"\bCould not\b", "Couldn't"),
            (r"\bshould not\b", "shouldn't"),
            (r"\bShould not\b", "Shouldn't"),
            (r"\bmust not\b", "mustn't"),
            (r"\bMust not\b", "Mustn't"),
            (r"\bis not\b", "isn't"),
            (r"\bIs not\b", "Isn't"),
            (r"\bare not\b", "aren't"),
            (r"\bAre not\b", "Aren't"),
            (r"\bwas not\b", "wasn't"),
            (r"\bWas not\b", "Wasn't"),
            (r"\bwere not\b", "weren't"),
            (r"\bWere not\b", "Weren't"),
            (r"\bhave not\b", "haven't"),
            (r"\bHave not\b", "Haven't"),
            (r"\bhas not\b", "hasn't"),
            (r"\bHas not\b", "Hasn't"),
            (r"\bhad not\b", "hadn't"),
            (r"\bHad not\b", "Hadn't"),
            (r"\bdoes not\b", "doesn't"),
            (r"\bDoes not\b", "Doesn't"),
            (r"\bdid not\b", "didn't"),
            (r"\bDid not\b", "Didn't"),
            (r"\bI am\b", "I'm"),
            (r"\bI have\b", "I've"),
            (r"\bI had\b", "I'd"),
            (r"\bI will\b", "I'll"),
            (r"\bI would\b", "I'd"),
            (r"\byou are\b", "you're"),
            (r"\bYou are\b", "You're"),
            (r"\byou have\b", "you've"),
            (r"\bYou have\b", "You've"),
            (r"\byou will\b", "you'll"),
            (r"\bYou will\b", "You'll"),
            (r"\byou would\b", "you'd"),
            (r"\bYou would\b", "You'd"),
            (r"\bthey are\b", "they're"),
            (r"\bThey are\b", "They're"),
            (r"\bthey have\b", "they've"),
            (r"\bThey have\b", "They've"),
            (r"\bthey will\b", "they'll"),
            (r"\bThey will\b", "They'll"),
            (r"\bthey would\b", "they'd"),
            (r"\bThey would\b", "They'd"),
            (r"\bwe are\b", "we're"),
            (r"\bWe are\b", "We're"),
            (r"\bwe have\b", "we've"),
            (r"\bWe have\b", "We've"),
            (r"\bwe will\b", "we'll"),
            (r"\bWe will\b", "We'll"),
            (r"\bwe would\b", "we'd"),
            (r"\bWe would\b", "We'd"),
            (r"\bit is\b", "it's"),
            (r"\bIt is\b", "It's"),
            (r"\bit has\b", "it's"),
            (r"\bIt has\b", "It's"),
            (r"\bit will\b", "it'll"),
            (r"\bIt will\b", "It'll"),
            (r"\bit would\b", "it'd"),
            (r"\bthat is\b", "that's"),
            (r"\bThat is\b", "That's"),
            (r"\bthat will\b", "that'll"),
            (r"\bThat will\b", "That'll"),
            (r"\bthat would\b", "that'd"),
            (r"\bwhat is\b", "what's"),
            (r"\bWhat is\b", "What's"),
            (r"\bwho is\b", "who's"),
            (r"\bWho is\b", "Who's"),
            (r"\bwho has\b", "who's"),
            (r"\bWho has\b", "Who's"),
            (r"\bwhere is\b", "where's"),
            (r"\bWhere is\b", "Where's"),
            (r"\bwhen is\b", "when's"),
            (r"\bWhen is\b", "When's"),
            (r"\bhow is\b", "how's"),
            (r"\bHow is\b", "How's"),
            (r"\bthere is\b", "there's"),
            (r"\bThere is\b", "There's"),
            (r"\bthere are\b", "there're"),
            (r"\bthere will\b", "there'll"),
            (r"\bhere is\b", "here's"),
            (r"\bHere is\b", "Here's"),
            (r"\blet us\b", "let's"),
            (r"\bLet us\b", "Let's"),
            (r"\bhe is\b", "he's"),
            (r"\bHe is\b", "He's"),
            (r"\bhe has\b", "he's"),
            (r"\bhe will\b", "he'll"),
            (r"\bhe would\b", "he'd"),
            (r"\bshe is\b", "she's"),
            (r"\bShe is\b", "She's"),
            (r"\bshe has\b", "she's"),
            (r"\bshe will\b", "she'll"),
            (r"\bshe would\b", "she'd"),
        ]

        result = text
        for pattern, replacement in contractions:
            result = re.sub(pattern, replacement, result)

        return result

    def _inject_burstiness(self, text: str) -> str:
        """Stage 5: Vary sentence lengths dramatically."""
        sentences = self._split_sentences(text)
        if len(sentences) < 3:
            return text

        result_sentences = []

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            words = sentence.split()
            word_count = len(words)

            # Sometimes split long sentences
            if word_count > 25 and random.random() < 0.4:
                # Find a good split point
                mid = word_count // 2
                for j in range(mid - 3, mid + 3):
                    if j < word_count and words[j].lower() in ['and', 'but', 'so', 'which', 'that', 'because', 'while', 'when']:
                        first_part = ' '.join(words[:j])
                        second_part = ' '.join(words[j+1:])
                        if first_part and second_part:
                            # Capitalize second part
                            second_part = second_part[0].upper() + second_part[1:] if second_part else second_part
                            result_sentences.append(first_part + '.')
                            result_sentences.append(second_part)
                            break
                else:
                    result_sentences.append(sentence)

            # Sometimes merge short sentences
            elif word_count < 8 and i < len(sentences) - 1 and random.random() < 0.3:
                next_sentence = sentences[i + 1].strip() if i + 1 < len(sentences) else ""
                if next_sentence and len(next_sentence.split()) < 15:
                    # Merge with comma or "and" - NO DASHES (AI signal!)
                    connector = random.choice([", ", ", and ", " and "])
                    merged = sentence.rstrip('.!?') + connector + next_sentence[0].lower() + next_sentence[1:]
                    result_sentences.append(merged)
                    sentences[i + 1] = ""  # Mark as processed
                else:
                    result_sentences.append(sentence)
            else:
                result_sentences.append(sentence)

        return ' '.join(result_sentences)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _add_human_quirks(self, text: str) -> str:
        """Stage 6: Add human writing quirks."""
        sentences = self._split_sentences(text)
        result_sentences = []

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            # 15% chance: Add a casual sentence starter
            if i > 0 and random.random() < 0.15:
                starter = random.choice(CASUAL_TRANSITIONS)
                if not any(sentence.startswith(s) for s in CASUAL_TRANSITIONS + HUMAN_STARTERS):
                    sentence = starter + sentence[0].lower() + sentence[1:]

            # 10% chance: Add parenthetical aside
            if random.random() < 0.10 and len(sentence.split()) > 8:
                aside = random.choice(PARENTHETICAL_ASIDES)
                words = sentence.split()
                insert_pos = random.randint(len(words)//3, 2*len(words)//3)
                words.insert(insert_pos, aside)
                sentence = ' '.join(words)

            # 8% chance: Insert filler word
            if random.random() < 0.08 and len(sentence.split()) > 5:
                filler = random.choice(HUMAN_FILLERS)
                words = sentence.split()
                insert_pos = random.randint(1, min(3, len(words)-1))
                words.insert(insert_pos, filler + ",")
                sentence = ' '.join(words)

            # 5% chance: Add rhetorical question
            if random.random() < 0.05 and i > 0:
                questions = [
                    "Right?",
                    "Makes sense?",
                    "See what I mean?",
                    "You know?",
                    "Pretty cool, huh?",
                    "Interesting, right?",
                ]
                sentence = sentence + " " + random.choice(questions)

            # 5% chance: Add a very short punchy sentence before
            if random.random() < 0.05 and i > 0:
                punchy = random.choice([
                    "Here's the thing.",
                    "Think about it.",
                    "Big difference.",
                    "Makes sense.",
                    "Simple enough.",
                    "Key point here.",
                    "Worth noting.",
                    "Quick aside.",
                ])
                result_sentences.append(punchy)

            result_sentences.append(sentence)

        return ' '.join(result_sentences)

    def _final_cleanup(self, text: str) -> str:
        """Stage 7: Final cleanup and polish."""
        result = text

        # IMPORTANT: Remove dashes - AI detection signal!
        # Replace em-dashes and en-dashes with commas or periods
        result = re.sub(r'\s*—\s*', ', ', result)  # em-dash
        result = re.sub(r'\s*–\s*', ', ', result)  # en-dash
        result = re.sub(r'\s+-\s+', ', ', result)  # spaced hyphen used as dash

        # Fix double commas that might result
        result = re.sub(r',\s*,', ',', result)

        # Fix double spaces
        result = re.sub(r'\s+', ' ', result)

        # Fix space before punctuation
        result = re.sub(r'\s+([.,!?;:])', r'\1', result)

        # Fix multiple punctuation
        result = re.sub(r'\.+', '.', result)
        result = re.sub(r'\?+', '?', result)
        result = re.sub(r'!+', '!', result)

        # Fix capitalization after periods
        def capitalize_after_period(match):
            return match.group(1) + ' ' + match.group(2).upper()
        result = re.sub(r'([.!?])\s+([a-z])', capitalize_after_period, result)

        # Ensure proper ending
        result = result.strip()
        if result and result[-1] not in '.!?':
            result += '.'

        return result


class HumanizerManager:
    """Manages the humanization workflow."""

    def __init__(self, humanizer: Humanizer):
        self.humanizer = humanizer
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def reset(self):
        self._cancelled = False

    def process_text(
        self,
        text: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Process text through the humanizer."""
        self._cancelled = False
        return self.humanizer.humanize(text, progress_callback)
