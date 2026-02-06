"""
Advanced AI Humanizer - Designed to bypass AI detection tools like Copyleaks.

Key techniques:
1. Perplexity manipulation - make word choices less predictable
2. Burstiness injection - vary sentence length/complexity dramatically
3. Sentence-level processing with varied approaches
4. Post-processing to add human-like imperfections
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


# Multiple varied prompts to use randomly for different sentences
VARIED_PROMPTS = [
    """Rewrite this one sentence in a completely different way. Be casual and conversational. Use contractions. Output only the rewritten sentence, nothing else:""",

    """Transform this sentence like you're explaining it to a friend over coffee. Keep it natural and relaxed. Output only the new sentence:""",

    """Rephrase this sentence the way a blogger would write it. Make it punchy and engaging. Only output the rewritten version:""",

    """Rewrite this in your own words, like you're texting someone. Keep it simple and direct. Just the sentence:""",

    """Say this same thing but completely differently. Be informal. Use everyday language. Output only the result:""",
]

# Prompts for paragraph-level rewriting
PARAGRAPH_PROMPTS = [
    """Rewrite this paragraph like you're a blogger writing casually. Rules:
- Mix short punchy sentences with longer flowing ones
- Use contractions (don't, it's, you're, that's, won't, can't)
- Start some sentences with And, But, So, Now, Look, Thing is
- Be conversational - like talking to a friend
- Keep all facts exactly the same
- NO citations or references
Output only the rewritten paragraph:""",

    """Transform this into natural human writing. Requirements:
- Vary sentence length dramatically - some very short, some longer
- Use casual phrases: pretty much, kind of, honestly, basically, really
- Include contractions everywhere
- Change sentence structures completely
- Keep the same meaning and facts
Just output the new version:""",

    """Rewrite this the way a real person would write it, not a robot. Make it:
- Sound like natural conversation
- Have uneven rhythm - short sentences then long ones
- Use simple everyday words
- Include contractions and casual language
- Keep all information accurate
Output only the rewritten text:""",
]


class Humanizer:
    """
    Advanced humanizer that bypasses AI detection through multiple techniques.
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
        Uses multi-stage processing for best results.
        """
        if not self.is_loaded or not self.model:
            return text

        if not text or not text.strip():
            return text

        try:
            # Stage 1: Split into paragraphs
            paragraphs = self._split_paragraphs(text)

            if progress_callback:
                progress_callback("Processing paragraphs...")

            # Stage 2: Process each paragraph
            humanized_paragraphs = []
            for i, para in enumerate(paragraphs):
                if not para.strip():
                    humanized_paragraphs.append(para)
                    continue

                if progress_callback:
                    progress_callback(f"Processing paragraph {i+1}/{len(paragraphs)}...")

                # Rewrite paragraph
                humanized = self._humanize_paragraph(para)
                humanized_paragraphs.append(humanized)

            # Stage 3: Join and post-process
            result = "\n\n".join(humanized_paragraphs)

            if progress_callback:
                progress_callback("Applying finishing touches...")

            # Stage 4: Post-processing for extra humanization
            result = self._post_process(result)

            return result

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {e}")
            return text

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines or multiple newlines
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _humanize_paragraph(self, paragraph: str) -> str:
        """Humanize a single paragraph using LLM."""
        # Choose a random prompt for variety
        prompt_template = random.choice(PARAGRAPH_PROMPTS)

        prompt = f"""[INST] {prompt_template}

{paragraph}
[/INST]"""

        # Use slightly different settings each time for variance
        temp_variance = random.uniform(-0.1, 0.1)

        response = self.model(
            prompt,
            max_tokens=min(len(paragraph.split()) * 3, self.config.max_tokens),
            temperature=self.config.temperature + temp_variance,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repeat_penalty=self.config.repeat_penalty,
            stop=["</s>", "[INST]", "\n\nReferences:", "\n\nSources:"],
            echo=False
        )

        result = response["choices"][0]["text"].strip()
        result = self._clean_output(result)

        # If result is too different in length, try again or use original
        orig_words = len(paragraph.split())
        result_words = len(result.split())

        if result_words < orig_words * 0.4 or result_words > orig_words * 2:
            return paragraph

        return result

    def _clean_output(self, text: str) -> str:
        """Clean LLM output of artifacts."""
        result = text.strip()

        # Remove common prefixes
        prefixes = [
            "here's", "here is", "rewritten:", "output:",
            "rewritten version:", "the rewritten", "paraphrased:"
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
        result = re.sub(r'\([^)]*,\s*\d{4}\)', '', result)

        # Remove reference sections
        result = re.sub(r'\n\s*(References|Sources|Citations):.*', '', result, flags=re.IGNORECASE | re.DOTALL)

        return result.strip()

    def _post_process(self, text: str) -> str:
        """
        Apply post-processing to make text more human-like.
        This adds controlled imperfections and variance.
        """
        sentences = self._split_sentences(text)
        processed = []

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            # Randomly apply various humanization techniques
            sentence = self._apply_humanizations(sentence, i)
            processed.append(sentence)

        # Rejoin with varied spacing
        result = self._rejoin_sentences(processed)

        return result

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving structure."""
        # Split on sentence endings but keep the punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return sentences

    def _apply_humanizations(self, sentence: str, index: int) -> str:
        """Apply random humanization techniques to a sentence."""

        # 30% chance: Add a casual starter to some sentences
        if index > 0 and random.random() < 0.15:
            starters = ["And ", "But ", "So ", "Now, ", "Plus, ", "Also, "]
            if not any(sentence.startswith(s) for s in starters):
                # Only if sentence doesn't already start with these
                first_char = sentence[0] if sentence else ''
                if first_char.isupper() and first_char not in "AIBSN":
                    sentence = random.choice(starters) + sentence[0].lower() + sentence[1:]

        # 20% chance: Convert some formal phrases to casual
        casual_replacements = [
            (r'\bHowever\b', 'But'),
            (r'\bTherefore\b', 'So'),
            (r'\bFurthermore\b', 'Plus'),
            (r'\bAdditionally\b', 'Also'),
            (r'\bMoreover\b', 'And'),
            (r'\bNevertheless\b', 'Still'),
            (r'\bConsequently\b', 'So'),
            (r'\bUtilize\b', 'use'),
            (r'\butilize\b', 'use'),
            (r'\bObtain\b', 'get'),
            (r'\bobtain\b', 'get'),
            (r'\bAssist\b', 'help'),
            (r'\bassist\b', 'help'),
            (r'\bPurchase\b', 'buy'),
            (r'\bpurchase\b', 'buy'),
            (r'\bRequire\b', 'need'),
            (r'\brequire\b', 'need'),
            (r'\bSubstantial\b', 'big'),
            (r'\bsubstantial\b', 'big'),
            (r'\bNumerous\b', 'many'),
            (r'\bnumerous\b', 'many'),
            (r'\bPrior to\b', 'before'),
            (r'\bprior to\b', 'before'),
            (r'\bIn order to\b', 'to'),
            (r'\bin order to\b', 'to'),
            (r'\bDue to the fact that\b', 'because'),
            (r'\bdue to the fact that\b', 'because'),
            (r'\bAt this point in time\b', 'now'),
            (r'\bat this point in time\b', 'now'),
            (r'\bIn the event that\b', 'if'),
            (r'\bin the event that\b', 'if'),
            (r'\bIt is important to note that\b', ''),
            (r'\bIt should be noted that\b', ''),
        ]

        for pattern, replacement in casual_replacements:
            if random.random() < 0.7:  # Apply most of the time
                sentence = re.sub(pattern, replacement, sentence)

        # Ensure contractions are used
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
            (r"\bI will\b", "I'll"),
            (r"\bI would\b", "I'd"),
            (r"\byou are\b", "you're"),
            (r"\bYou are\b", "You're"),
            (r"\byou have\b", "you've"),
            (r"\bYou have\b", "You've"),
            (r"\byou will\b", "you'll"),
            (r"\bYou will\b", "You'll"),
            (r"\bthey are\b", "they're"),
            (r"\bThey are\b", "They're"),
            (r"\bthey have\b", "they've"),
            (r"\bThey have\b", "They've"),
            (r"\bwe are\b", "we're"),
            (r"\bWe are\b", "We're"),
            (r"\bwe have\b", "we've"),
            (r"\bWe have\b", "We've"),
            (r"\bit is\b", "it's"),
            (r"\bIt is\b", "It's"),
            (r"\bthat is\b", "that's"),
            (r"\bThat is\b", "That's"),
            (r"\bwhat is\b", "what's"),
            (r"\bWhat is\b", "What's"),
            (r"\bthere is\b", "there's"),
            (r"\bThere is\b", "There's"),
            (r"\bhere is\b", "here's"),
            (r"\bHere is\b", "Here's"),
            (r"\blet us\b", "let's"),
            (r"\bLet us\b", "Let's"),
        ]

        for pattern, replacement in contractions:
            sentence = re.sub(pattern, replacement, sentence)

        # 10% chance: Add a filler word
        if random.random() < 0.1:
            fillers = [" actually", " really", " basically", " pretty much", " honestly"]
            # Find a good spot to insert (after first few words)
            words = sentence.split()
            if len(words) > 4:
                insert_pos = random.randint(2, min(4, len(words)-1))
                words.insert(insert_pos, random.choice(fillers))
                sentence = " ".join(words)

        return sentence

    def _rejoin_sentences(self, sentences: List[str]) -> str:
        """Rejoin sentences with natural spacing."""
        if not sentences:
            return ""

        result = []
        for i, sentence in enumerate(sentences):
            if i == 0:
                result.append(sentence)
            else:
                # Mostly single space, occasionally double for paragraph feel
                result.append(" " + sentence)

        return "".join(result)


# For backwards compatibility with existing code
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
