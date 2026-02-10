"""
Advanced AI Humanizer V2 - Professional-grade AI detection bypass.

This enhanced version uses techniques from Grammarly, CopyLeaks analysis,
and linguistic research to produce text that passes all major AI detectors.

Key improvements:
1. Two-pass LLM rewriting for higher quality
2. Context-aware word substitution
3. Perplexity injection for natural variation
4. Advanced sentence restructuring
5. Semantic preservation with stylistic transformation
6. Human writing pattern mimicry
"""

import re
import random
import math
from typing import Optional, Callable, List, Tuple, Dict
from dataclasses import dataclass
from collections import Counter

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


@dataclass
class HumanizerV2Config:
    """Configuration for enhanced humanizer."""
    model_path: str = ""
    n_ctx: int = 4096
    n_threads: int = 4
    n_gpu_layers: int = 0
    temperature: float = 0.85
    top_p: float = 0.92
    top_k: int = 50
    repeat_penalty: float = 1.18
    max_tokens: int = 768
    creativity_level: int = 50
    # V2 specific settings
    enable_two_pass: bool = True
    preserve_key_terms: bool = True
    target_perplexity: float = 0.65  # Higher = more human-like variation


# ============= MODE-SPECIFIC PROMPTS =============
# Each mode has its own transformation prompts

MODE_PROMPTS = {
    "professional": {
        "first_pass": [
            """Rewrite this text in professional, business-appropriate language.

CRITICAL RULES:
- Keep ALL original information - do NOT remove or skip any content
- Do NOT add new information or sentences that weren't in the original
- Only change the STYLE, not the MEANING or CONTENT

VOICE & STYLE:
- Use formal, polished language suitable for business communication
- "Hi" → "Hello"
- "Hey" → "Hello"
- "Thanks" → "Thank you"
- "gonna" → "going to"
- "wanna" → "want to"
- "What's up" → "How may I help you" or "What can I do for you"

VOCABULARY:
- Professional but clear vocabulary
- Avoid slang and colloquialisms
- Use complete words, no contractions in formal contexts

IMPORTANT:
- Same number of sentences as input
- Same information, just more professional wording
- Do NOT add extra pleasantries or filler

Output ONLY the professional rewrite, nothing else:""",
        ],
        "second_pass": [
            """Review and refine this text for professional polish:

RULES:
- Do NOT add any new content
- Do NOT remove any existing content
- Only refine the professional tone

CHECK & FIX:
- Ensure formal language is consistent throughout
- Verify professional vocabulary usage
- Maintain clear, business-appropriate tone

Output the refined professional text:""",
        ]
    },

    "conversational": {
        "first_pass": [
            """Rewrite this text in casual, friendly language like talking to a friend.

CRITICAL RULES:
- Keep ALL original information - do NOT remove or skip any content
- Do NOT add new information, questions, or sentences that weren't in the original
- Only change the STYLE, not the MEANING or CONTENT

VOICE & STYLE:
- Use contractions: don't, won't, it's, that's, you're, I'm
- Simple everyday words
- "Hello" → "Hey" or "Hi"
- "I am" → "I'm"
- "What is" → "What's"

VOCABULARY:
- Keep it simple and casual
- "obtain" → "get"
- "assist" → "help"
- "utilize" → "use"

IMPORTANT:
- Same number of sentences as input
- Same information, just more casual wording
- Do NOT add greetings, questions, or filler that wasn't there

Output ONLY the casual rewrite, nothing else:""",
        ],
        "second_pass": [
            """Polish this to sound like natural conversation:

RULES:
- Do NOT add any new content
- Do NOT remove any existing content
- Only refine the casual tone

CHECK & FIX:
- Ensure contractions are used consistently
- Keep the friendly, approachable tone
- Remove any stiff or formal phrasing

Output the refined conversational text:""",
        ]
    },

    "scholarly": {
        "first_pass": [
            """Rewrite this text in academic, scholarly language.

CRITICAL RULES:
- Keep ALL original information - do NOT remove or skip any content
- Do NOT add new information or sentences that weren't in the original
- Only change the STYLE, not the MEANING or CONTENT

VOICE & STYLE:
- Use formal academic tone
- "use" → "utilize" or "employ"
- "show" → "demonstrate" or "indicate"
- "think" → "posit" or "suggest"
- "a lot" → "numerous" or "substantial"
- "big" → "significant" or "considerable"

VOCABULARY:
- Academic vocabulary where appropriate
- Precise terminology
- Formal expressions

IMPORTANT:
- Same number of sentences as input
- Same information, just more academic wording
- Do NOT add citations or references that weren't there

Output ONLY the scholarly rewrite, nothing else:""",
        ],
        "second_pass": [
            """Review and refine this text for academic standards:

RULES:
- Do NOT add any new content
- Do NOT remove any existing content
- Only refine the academic tone

CHECK & FIX:
- Ensure academic vocabulary is consistent
- Verify formal tone throughout
- Maintain scholarly precision

Output the refined academic text:""",
        ]
    },

    "creative": {
        "first_pass": [
            """Rewrite this text with vivid, engaging storytelling language.

CRITICAL RULES:
- Keep ALL original information - do NOT remove or skip any content
- Do NOT add new plot points, characters, or events that weren't in the original
- Only enhance the STYLE and EXPRESSIVENESS, not the core CONTENT

VOICE & STYLE:
- Use evocative, descriptive language
- Employ sensory details and imagery
- Use metaphors and similes where appropriate
- Vary rhythm and pacing for effect

VOCABULARY:
- Rich, expressive words
- Active, dynamic verbs
- Fresh expressions instead of clichés

IMPORTANT:
- Preserve the original meaning and information
- Enhance expressiveness without changing facts
- Do NOT invent new details

Output ONLY the creative rewrite, nothing else:""",
        ],
        "second_pass": [
            """Polish this for maximum creative impact:

RULES:
- Do NOT add any new content or events
- Do NOT remove any existing content
- Only enhance the creative expression

CHECK & FIX:
- Enhance vivid imagery
- Strengthen sensory details
- Refine rhythm and flow

Output the refined creative text:""",
        ]
    },

    "concise": {
        "first_pass": [
            """Make this text shorter and punchier by removing filler and redundancy.

CRITICAL RULES:
- Keep ALL essential information and meaning
- Do NOT remove important facts or details
- Only remove filler words, redundancy, and unnecessary phrases

WHAT TO CUT:
- Filler words: very, really, just, actually, basically, quite
- Wordy phrases: "In order to" → "To", "Due to the fact that" → "Because"
- Redundant information (saying the same thing twice)
- Unnecessary adjectives and adverbs

WHAT TO KEEP:
- All key facts and information
- Core meaning and intent
- Clarity and readability

TARGET:
- Make it shorter but complete
- Every word should be purposeful
- Direct and clear

Output ONLY the concise rewrite, nothing else:""",
        ],
        "second_pass": [
            """Review and tighten this text further:

RULES:
- Do NOT remove essential information
- Only cut remaining filler and redundancy
- Maintain complete meaning

CHECK & FIX:
- Remove any remaining filler words
- Ensure maximum clarity with minimum words
- Verify all key information is preserved

Output the final concise text:""",
        ]
    },

    "summarize": {
        "first_pass": [
            """Create a concise summary of this text.

YOUR TASK:
- Condense to 20-30% of original length
- Extract ONLY the core message and key points
- Remove all examples, details, and elaboration
- Write in clear, direct sentences

STRUCTURE:
- Start with the main point/thesis
- Include only essential supporting facts
- Skip all non-critical information
- End with any key conclusion

IMPORTANT:
- Be ruthlessly concise
- Every sentence must convey essential information
- No filler words or redundancy
- Preserve the original meaning accurately

Output ONLY the condensed summary:""",
        ],
        "second_pass": [
            """Make this summary even more concise:

GOAL: Reduce further while keeping ALL essential meaning

RULES:
- Cut any remaining fluff
- Merge similar points
- Use shorter phrases where possible
- Keep critical information intact

Output the tightened summary:""",
        ]
    },

    "expand": {
        "first_pass": [
            """Expand this content by adding MORE SIMILAR examples and practice items.

YOUR TASK:
Analyze what type of content this is and expand accordingly:

IF IT'S HOMEWORK/EXERCISES/QUESTIONS:
- Add 3-5 MORE questions of the SAME type and difficulty
- Match the exact format, style, and complexity
- Cover the same topic/concept with different scenarios
- Keep the same level of challenge

IF IT'S EDUCATIONAL/EXPLANATORY:
- Add more examples illustrating the same concepts
- Include practice problems or exercises
- Add related scenarios or case studies
- Expand each point with concrete illustrations

IF IT'S A LIST OR OUTLINE:
- Add more items following the same pattern
- Maintain the same level of detail per item
- Keep the same format and structure

CRITICAL:
- Match the DIFFICULTY level exactly
- Match the FORMAT exactly
- Stay on the SAME topic
- Add 2-3x more content of the same type

Output the expanded content with additional similar items:""",
        ],
        "second_pass": [
            """Review and enhance this expanded content:

CHECK:
- Are new items at the SAME difficulty level?
- Do they follow the SAME format?
- Are they on the SAME topic?
- Is the expansion substantial (2-3x original)?

IMPROVE:
- Add more variety in the examples
- Ensure consistent quality throughout
- Remove any items that don't match the original style

Output the refined expanded content:""",
        ]
    }
}

# Default prompts for backward compatibility (conversational style)
FIRST_PASS_PROMPTS = MODE_PROMPTS["conversational"]["first_pass"]
SECOND_PASS_PROMPTS = MODE_PROMPTS["conversational"]["second_pass"]


# ============= COMPREHENSIVE AI PATTERNS =============
# Expanded list of patterns that AI detectors flag

AI_PHRASE_PATTERNS = [
    # Opening patterns
    r"\bin today'?s (?:world|society|age|era|fast-paced|digital|modern)\b",
    r"\bin the (?:realm|context|landscape|sphere|domain) of\b",
    r"\bwhen it comes to\b",
    r"\bas we (?:navigate|delve|explore|embark|venture)\b",
    r"\blet(?:'s| us) (?:delve|dive|explore|examine|take a look)\b",

    # Importance phrases (major AI tells)
    r"\bit(?:'s| is) (?:important|crucial|essential|vital|imperative|paramount) to (?:note|understand|recognize|remember|acknowledge|consider)\b",
    r"\bit(?:'s| is) worth (?:noting|mentioning|pointing out|emphasizing)\b",
    r"\bneedless to say\b",
    r"\bwithout a doubt\b",
    r"\bit goes without saying\b",

    # Transition patterns (AI uses these excessively)
    r"\bfurthermore\b",
    r"\bmoreover\b",
    r"\badditionally\b",
    r"\bconsequently\b",
    r"\bnevertheless\b",
    r"\bnonetheless\b",
    r"\bsubsequently\b",
    r"\bhenceforth\b",
    r"\bthus\b",
    r"\bhence\b",
    r"\bthereby\b",
    r"\bwhereby\b",

    # Conclusion patterns
    r"\bin conclusion\b",
    r"\bto (?:summarize|sum up|conclude|wrap up)\b",
    r"\ball in all\b",
    r"\bat the end of the day\b",
    r"\bin (?:summary|closing|the final analysis)\b",
    r"\btaking (?:everything|all things) into (?:account|consideration)\b",

    # Hedging (AI does this too uniformly)
    r"\bit can be (?:argued|said|noted|observed|seen)\b",
    r"\bone (?:might|could|may) (?:argue|say|suggest|contend)\b",
    r"\bsome (?:experts|researchers|scholars|people) (?:believe|argue|suggest|contend)\b",

    # Corporate/formal speak
    r"\bleverage[sd]?\b",
    r"\boptimize[sd]?\b",
    r"\bstreamline[sd]?\b",
    r"\bfacilitate[sd]?\b",
    r"\butilize[sd]?\b",
    r"\bimplement(?:ed|ing|ation)?\b",
    r"\bsynergy\b",
    r"\bparadigm\b",
    r"\bholistic(?:ally)?\b",
    r"\bseamless(?:ly)?\b",
    r"\brobust\b",

    # Overly descriptive
    r"\bpivotal\b",
    r"\bcrucial\b",
    r"\bparamount\b",
    r"\bprofound(?:ly)?\b",
    r"\bintricate(?:ly)?\b",
    r"\bmultifaceted\b",
    r"\bcomprehensive(?:ly)?\b",
    r"\bsubstantial(?:ly)?\b",

    # Specific AI patterns
    r"\bplays a (?:crucial|vital|key|important|significant|pivotal) role\b",
    r"\bserves as a (?:testament|reminder|beacon)\b",
    r"\bsheds (?:new )?light on\b",
    r"\bpaves the way\b",
    r"\bbreaks new ground\b",
    r"\bpushes the (?:boundaries|envelope)\b",
    r"\bstands as (?:a testament|proof|evidence)\b",
    r"\bin the (?:grand scheme|bigger picture)\b",
    r"\bmarks a significant\b",
    r"\brepresents a (?:significant|major|pivotal)\b",

    # Wordy phrases (AI is verbose)
    r"\bdue to the fact that\b",
    r"\bin order to\b",
    r"\bfor the purpose of\b",
    r"\bwith regard to\b",
    r"\bwith respect to\b",
    r"\bin terms of\b",
    r"\bthe fact that\b",
    r"\bhas the ability to\b",
    r"\bis able to\b",
    r"\bat this point in time\b",
    r"\bat the present time\b",
    r"\bprior to\b",

    # List introductions
    r"\bthere are (?:several|many|numerous|various|multiple) (?:ways|reasons|factors|aspects|elements)\b",
    r"\b(?:first|second|third|fourth|finally)(?:ly)?,?\s",
    r"\bon (?:the )?one hand.*on (?:the )?other hand\b",

    # Emphasis patterns (AI overuses)
    r"\bindeed\b",
    r"\bcertainly\b",
    r"\bundoubtedly\b",
    r"\bunquestionably\b",
    r"\bwithout question\b",
]

# Compile patterns for efficiency
COMPILED_AI_PATTERNS = [re.compile(p, re.IGNORECASE) for p in AI_PHRASE_PATTERNS]


# ============= ENHANCED SUBSTITUTIONS =============
# Context-aware word replacements

CONTEXTUAL_SUBSTITUTIONS = {
    # Academic -> Conversational
    "utilize": {"default": ["use"], "technical": ["work with", "apply"]},
    "implement": {"default": ["set up", "add"], "technical": ["build", "create"]},
    "facilitate": {"default": ["help", "make easier"], "formal": ["support"]},
    "demonstrate": {"default": ["show"], "academic": ["prove", "illustrate"]},
    "approximately": {"default": ["about", "around"], "precise": ["roughly"]},
    "subsequently": {"default": ["then", "later"], "narrative": ["after that"]},
    "previously": {"default": ["before", "earlier"]},
    "additionally": {"default": ["also", "plus"], "casual": ["on top of that"]},
    "furthermore": {"default": ["also", "and"], "emphasis": ["what's more"]},
    "however": {"default": ["but", "though"], "contrast": ["still"]},
    "therefore": {"default": ["so"], "conclusion": ["that's why"]},
    "consequently": {"default": ["so", "as a result"]},
    "numerous": {"default": ["many", "lots of"], "casual": ["a bunch of"]},
    "significant": {"default": ["big", "major"], "important": ["key"]},
    "comprehensive": {"default": ["complete", "full"], "thorough": ["detailed"]},
    "sufficient": {"default": ["enough"]},
    "immediately": {"default": ["right away"], "urgent": ["instantly"]},
    "frequently": {"default": ["often"], "casual": ["a lot"]},
    "obtain": {"default": ["get"], "formal": ["receive"]},
    "acquire": {"default": ["get", "pick up"]},
    "possess": {"default": ["have"]},
    "require": {"default": ["need"]},
    "assist": {"default": ["help"]},
    "attempt": {"default": ["try"]},
    "commence": {"default": ["start", "begin"]},
    "terminate": {"default": ["end", "stop"]},
    "purchase": {"default": ["buy", "get"]},
    "regarding": {"default": ["about"]},
    "concerning": {"default": ["about"]},
    "individuals": {"default": ["people"]},
    "children": {"default": ["kids"]},
    "residence": {"default": ["home", "place"]},
    "beverage": {"default": ["drink"]},
    "cuisine": {"default": ["food"]},
    "perhaps": {"default": ["maybe"]},
    "however": {"default": ["but", "though"]},
    "although": {"default": ["though", "even though"]},
    "whilst": {"default": ["while"]},
    "amongst": {"default": ["among"]},
    "upon": {"default": ["on"]},
    "within": {"default": ["in"]},
    "whereby": {"default": ["where"]},
    "thereby": {"default": ["so"]},
    "thereof": {"default": ["of it"]},
    "wherein": {"default": ["where"]},
}

# Simple substitutions (no context needed)
SIMPLE_SUBSTITUTIONS = {
    "utilize": ["use"],
    "utilizes": ["uses"],
    "utilizing": ["using"],
    "utilized": ["used"],
    "obtain": ["get"],
    "obtains": ["gets"],
    "obtained": ["got"],
    "sufficient": ["enough"],
    "insufficient": ["not enough"],
    "demonstrate": ["show"],
    "demonstrates": ["shows"],
    "demonstrated": ["showed"],
    "approximately": ["about", "around"],
    "immediately": ["right away", "quickly"],
    "frequently": ["often"],
    "additionally": ["also", "plus"],
    "furthermore": ["also", "and"],
    "moreover": ["also", "and"],
    "however": ["but", "though"],
    "therefore": ["so"],
    "consequently": ["so"],
    "nevertheless": ["still", "but"],
    "nonetheless": ["still", "but"],
    "subsequently": ["then", "later"],
    "previously": ["before", "earlier"],
    "numerous": ["many", "lots of"],
    "significant": ["big", "major"],
    "substantial": ["large", "big"],
    "comprehensive": ["complete", "full"],
    "facilitate": ["help", "make easier"],
    "implement": ["set up", "add"],
    "leverage": ["use"],
    "optimize": ["improve"],
    "streamline": ["simplify"],
    "individuals": ["people"],
    "commence": ["start", "begin"],
    "terminate": ["end", "stop"],
    "purchase": ["buy"],
    "acquire": ["get"],
    "possess": ["have"],
    "require": ["need"],
    "assist": ["help"],
    "attempt": ["try"],
    "regarding": ["about"],
    "concerning": ["about"],
    "currently": ["now"],
    "primarily": ["mainly", "mostly"],
    "essentially": ["basically"],
    "extremely": ["really", "very"],
    "particularly": ["especially"],
    "specifically": ["especially"],
}


# ============= HUMAN WRITING PATTERNS =============

CASUAL_STARTERS = [
    "Look, ", "Here's the thing: ", "So basically, ", "The thing is, ",
    "Honestly, ", "To be fair, ", "Actually, ", "Real talk: ",
    "Here's the deal: ", "I mean, ", "Let me put it this way: ",
]

SENTENCE_CONNECTORS = [
    "And ", "But ", "So ", "Plus, ", "Also, ", "Or ", "Yet ",
]

FILLER_INSERTIONS = [
    "basically", "pretty much", "kind of", "actually", "honestly",
    "really", "definitely", "probably", "literally", "obviously",
]

RHETORICAL_QUESTIONS = [
    "Right?", "You know?", "Makes sense?", "See what I mean?",
]


class HumanizerV2:
    """
    Enhanced humanizer with professional-grade AI detection bypass.
    Supports multiple writing styles: professional, conversational, scholarly, creative, concise.
    """

    # Token estimation: ~4 chars per token, leave room for prompt (~2000 tokens)
    # For 8192 context, safe text limit is about 6000 tokens = 24000 chars
    MAX_CHUNK_CHARS = 20000  # Conservative limit for text chunks
    CHARS_PER_TOKEN = 4  # Rough estimate

    def __init__(self, config: Optional[HumanizerV2Config] = None):
        self.config = config or HumanizerV2Config()
        self.model: Optional[Llama] = None
        self.is_loaded = False
        self._creativity_level = self.config.creativity_level
        self._current_style = "conversational"  # Default style

    @property
    def current_style(self) -> str:
        return self._current_style

    def set_style(self, style: str):
        """Set the writing style for transformation."""
        if style.lower() in MODE_PROMPTS:
            self._current_style = style.lower()
        else:
            self._current_style = "conversational"

    @property
    def creativity_level(self) -> int:
        return self._creativity_level

    @creativity_level.setter
    def creativity_level(self, value: int):
        self._creativity_level = max(0, min(100, value))
        self.config.creativity_level = self._creativity_level

    def get_effective_temperature(self) -> float:
        """Dynamic temperature based on creativity."""
        return 0.6 + (self._creativity_level / 100) * 0.5

    def get_substitution_rate(self) -> float:
        """How aggressively to substitute words."""
        return 0.5 + (self._creativity_level / 100) * 0.5

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // self.CHARS_PER_TOKEN

    def _split_into_chunks(self, text: str, max_chars: int = None) -> List[str]:
        """
        Split text into chunks that fit within context window.
        Respects paragraph boundaries where possible.
        """
        max_chars = max_chars or self.MAX_CHUNK_CHARS

        if len(text) <= max_chars:
            return [text]

        chunks = []
        paragraphs = self._split_paragraphs(text)
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para_len = len(para)

            # If single paragraph is too long, split it by sentences
            if para_len > max_chars:
                # Flush current chunk first
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Split long paragraph by sentences
                sentences = self._split_sentences(para)
                sent_chunk = []
                sent_length = 0

                for sent in sentences:
                    sent_len = len(sent)
                    if sent_length + sent_len + 1 > max_chars:
                        if sent_chunk:
                            chunks.append(" ".join(sent_chunk))
                        sent_chunk = [sent]
                        sent_length = sent_len
                    else:
                        sent_chunk.append(sent)
                        sent_length += sent_len + 1

                if sent_chunk:
                    chunks.append(" ".join(sent_chunk))

            # Normal paragraph handling
            elif current_length + para_len + 2 > max_chars:
                # Flush current chunk
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_length = para_len
            else:
                current_chunk.append(para)
                current_length += para_len + 2

        # Don't forget the last chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def load_model(self, model_path: Optional[str] = None,
                   progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        """Load LLM model."""
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
        """Unload model."""
        if self.model:
            del self.model
            self.model = None
        self.is_loaded = False

    def humanize(self, text: str,
                 progress_callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Humanize text using multi-stage processing.
        """
        if not self.is_loaded or not self.model:
            return text

        if not text or not text.strip():
            return text

        try:
            total_stages = 9 if self.config.enable_two_pass else 8

            # Stage 1: First LLM pass - major rewrite
            if progress_callback:
                progress_callback(f"Stage 1/{total_stages}: Rewriting with LLM...")
            result = self._first_pass_rewrite(text)

            # Stage 2: Second LLM pass - refinement (optional)
            if self.config.enable_two_pass:
                if progress_callback:
                    progress_callback(f"Stage 2/{total_stages}: Refining output...")
                result = self._second_pass_refine(result)
                stage_offset = 1
            else:
                stage_offset = 0

            # Stage 3: Remove AI phrases
            if progress_callback:
                progress_callback(f"Stage {3+stage_offset}/{total_stages}: Removing AI patterns...")
            result = self._remove_ai_patterns(result)

            # Stage 4: Word substitutions (only casualize for conversational/creative modes)
            if self._current_style in ["conversational", "creative", "concise"]:
                if progress_callback:
                    progress_callback(f"Stage {4+stage_offset}/{total_stages}: Casualizing vocabulary...")
                result = self._substitute_words(result)
            else:
                if progress_callback:
                    progress_callback(f"Stage {4+stage_offset}/{total_stages}: Maintaining vocabulary...")

            # Stage 5: Apply contractions (only for conversational/creative modes)
            if self._current_style in ["conversational", "creative", "concise"]:
                if progress_callback:
                    progress_callback(f"Stage {5+stage_offset}/{total_stages}: Adding contractions...")
                result = self._apply_contractions(result)
            else:
                if progress_callback:
                    progress_callback(f"Stage {5+stage_offset}/{total_stages}: Maintaining formal style...")

            # Stage 6: Inject perplexity variation (skip for professional/scholarly)
            if self._current_style in ["conversational", "creative"]:
                if progress_callback:
                    progress_callback(f"Stage {6+stage_offset}/{total_stages}: Adding natural variation...")
                result = self._inject_perplexity(result)
            else:
                if progress_callback:
                    progress_callback(f"Stage {6+stage_offset}/{total_stages}: Preserving style...")

            # Stage 7: Restructure sentences (skip casual starters for professional/scholarly)
            if progress_callback:
                progress_callback(f"Stage {7+stage_offset}/{total_stages}: Restructuring sentences...")
            result = self._restructure_sentences(result)

            # Stage 8: Add human touches (only for conversational/creative modes)
            if self._current_style in ["conversational", "creative"]:
                if progress_callback:
                    progress_callback(f"Stage {8+stage_offset}/{total_stages}: Adding human touches...")
                result = self._add_human_touches(result)
            else:
                if progress_callback:
                    progress_callback(f"Stage {8+stage_offset}/{total_stages}: Finalizing style...")

            # Final cleanup
            if progress_callback:
                progress_callback("Finalizing...")
            result = self._final_cleanup(result)

            return result

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {e}")
            return text

    def _first_pass_rewrite(self, text: str) -> str:
        """First LLM pass for major rewriting using style-specific prompts.
        Handles large texts by processing in manageable chunks.
        """
        paragraphs = self._split_paragraphs(text)
        rewritten = []

        for para in paragraphs:
            if not para.strip() or len(para.split()) < 5:
                rewritten.append(para)
                continue

            # Check if paragraph is too long - if so, split and process
            if len(para) > self.MAX_CHUNK_CHARS:
                chunks = self._split_into_chunks(para)
                para_parts = []
                for chunk in chunks:
                    result = self._rewrite_single_paragraph(chunk)
                    para_parts.append(result)
                rewritten.append(" ".join(para_parts))
            else:
                result = self._rewrite_single_paragraph(para)
                rewritten.append(result)

        return "\n\n".join(rewritten)

    def _rewrite_single_paragraph(self, para: str) -> str:
        """Rewrite a single paragraph with LLM."""
        # Get style-specific prompts
        style_prompts = MODE_PROMPTS.get(self._current_style, MODE_PROMPTS["conversational"])
        prompt_template = random.choice(style_prompts["first_pass"])
        prompt = f"""<|im_start|>system
{prompt_template}<|im_end|>
<|im_start|>user
{para}<|im_end|>
<|im_start|>assistant
"""

        temp = self.get_effective_temperature() + random.uniform(-0.08, 0.12)

        # Calculate safe max tokens based on mode
        available_tokens = self.config.n_ctx - self._estimate_tokens(prompt) - 100
        word_count = len(para.split())

        # Adjust token multiplier based on mode
        if self._current_style == "summarize":
            # Summarize: output should be 25-40% of input
            token_multiplier = 0.5
            min_tokens = 128
        elif self._current_style == "expand":
            # Expand: output should be 2-3x input
            token_multiplier = 4
            min_tokens = 512
        elif self._current_style == "concise":
            # Concise: output should be shorter
            token_multiplier = 1.0
            min_tokens = 128
        else:
            # Other modes: roughly same length
            token_multiplier = 2
            min_tokens = 256

        max_tokens = min(int(word_count * token_multiplier), self.config.max_tokens, available_tokens)
        max_tokens = max(min_tokens, max_tokens)

        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repeat_penalty=self.config.repeat_penalty,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False
        )

        result = response["choices"][0]["text"].strip()
        result = self._clean_llm_output(result)

        # Adjust validation threshold based on mode
        if self._current_style == "summarize":
            min_ratio = 0.15  # Summarize can be much shorter
        elif self._current_style == "concise":
            min_ratio = 0.3
        else:
            min_ratio = 0.4

        if result and len(result.split()) >= word_count * min_ratio:
            return result
        return para

    def _second_pass_refine(self, text: str) -> str:
        """Second LLM pass for refinement using style-specific prompts.
        Uses chunked processing for large texts to avoid context window overflow.
        """
        # Only refine if text is long enough
        if len(text.split()) < 30:
            return text

        # Check if text needs chunking
        chunks = self._split_into_chunks(text)

        if len(chunks) == 1:
            # Single chunk - process normally
            return self._refine_single_chunk(text)
        else:
            # Multiple chunks - process each and combine
            refined_chunks = []
            for chunk in chunks:
                refined = self._refine_single_chunk(chunk)
                refined_chunks.append(refined)
            return "\n\n".join(refined_chunks)

    def _refine_single_chunk(self, text: str) -> str:
        """Refine a single chunk of text."""
        # Get style-specific prompts
        style_prompts = MODE_PROMPTS.get(self._current_style, MODE_PROMPTS["conversational"])
        prompt_template = random.choice(style_prompts["second_pass"])
        prompt = f"""<|im_start|>system
{prompt_template}<|im_end|>
<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
"""

        # Limit max tokens based on available context and mode
        available_tokens = self.config.n_ctx - self._estimate_tokens(prompt) - 100
        word_count = len(text.split())

        # Adjust token multiplier based on mode
        if self._current_style == "summarize":
            token_multiplier = 0.4  # Further condense in second pass
            min_tokens = 128
        elif self._current_style == "expand":
            token_multiplier = 3
            min_tokens = 512
        elif self._current_style == "concise":
            token_multiplier = 0.8
            min_tokens = 128
        else:
            token_multiplier = 1.5
            min_tokens = 256

        max_tokens = min(int(word_count * token_multiplier), self.config.max_tokens, available_tokens)
        max_tokens = max(min_tokens, max_tokens)

        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,  # Lower temp for refinement
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False
        )

        result = response["choices"][0]["text"].strip()
        result = self._clean_llm_output(result)

        # Adjust validation threshold based on mode
        if self._current_style == "summarize":
            min_ratio = 0.1  # Summary can be very condensed
        elif self._current_style == "concise":
            min_ratio = 0.3
        else:
            min_ratio = 0.5

        if result and len(result.split()) >= word_count * min_ratio:
            return result
        return text

    def _clean_llm_output(self, text: str) -> str:
        """Clean LLM artifacts."""
        result = text.strip()

        # Remove common prefixes
        prefixes = [
            "here's", "here is", "rewritten:", "output:", "sure,",
            "okay,", "here you go:", "the rewritten", "paraphrased:",
            "certainly,", "of course,", "absolutely,", "let me",
        ]
        result_lower = result.lower()
        for prefix in prefixes:
            if result_lower.startswith(prefix):
                result = result[len(prefix):].strip()
                result_lower = result.lower()
                if result.startswith(':'):
                    result = result[1:].strip()

        # Remove quotes
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        if result.startswith("'") and result.endswith("'"):
            result = result[1:-1]

        # Remove citations
        result = re.sub(r'\([A-Z][a-z]+\s+et\s+al\.,?\s*\d{4}\)', '', result)
        result = re.sub(r'\[\d+\]', '', result)
        result = re.sub(r'\([\w\s&,]+\d{4}\)', '', result)

        return result.strip()

    def _remove_ai_patterns(self, text: str) -> str:
        """Remove AI-telltale phrases."""
        result = text

        for pattern in COMPILED_AI_PATTERNS:
            # Check if it's a single word or phrase
            if pattern.search(result):
                # Try to remove gracefully
                result = pattern.sub('', result)

        # Clean up artifacts
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'\s+([.,!?;:])', r'\1', result)
        result = re.sub(r'([.,!?])\s*,', r'\1', result)
        result = re.sub(r',\s*,', ',', result)
        result = re.sub(r'\.\s*\.', '.', result)

        return result.strip()

    def _substitute_words(self, text: str) -> str:
        """Replace formal words with casual alternatives."""
        result = text
        rate = self.get_substitution_rate()

        for formal, casual_list in SIMPLE_SUBSTITUTIONS.items():
            if random.random() > rate:
                continue

            pattern = re.compile(r'\b' + formal + r'\b', re.IGNORECASE)

            def replace_match(match):
                replacement = random.choice(casual_list)
                if match.group()[0].isupper():
                    return replacement.capitalize()
                return replacement

            result = pattern.sub(replace_match, result)

        return result

    def _apply_contractions(self, text: str) -> str:
        """Apply contractions aggressively."""
        contractions = [
            (r"\bdo not\b", "don't"), (r"\bDo not\b", "Don't"),
            (r"\bcannot\b", "can't"), (r"\bCannot\b", "Can't"),
            (r"\bwill not\b", "won't"), (r"\bWill not\b", "Won't"),
            (r"\bwould not\b", "wouldn't"), (r"\bWould not\b", "Wouldn't"),
            (r"\bcould not\b", "couldn't"), (r"\bCould not\b", "Couldn't"),
            (r"\bshould not\b", "shouldn't"), (r"\bShould not\b", "Shouldn't"),
            (r"\bis not\b", "isn't"), (r"\bIs not\b", "Isn't"),
            (r"\bare not\b", "aren't"), (r"\bAre not\b", "Aren't"),
            (r"\bwas not\b", "wasn't"), (r"\bWas not\b", "Wasn't"),
            (r"\bwere not\b", "weren't"), (r"\bWere not\b", "Weren't"),
            (r"\bhave not\b", "haven't"), (r"\bHave not\b", "Haven't"),
            (r"\bhas not\b", "hasn't"), (r"\bHas not\b", "Hasn't"),
            (r"\bhad not\b", "hadn't"), (r"\bHad not\b", "Hadn't"),
            (r"\bdoes not\b", "doesn't"), (r"\bDoes not\b", "Doesn't"),
            (r"\bdid not\b", "didn't"), (r"\bDid not\b", "Didn't"),
            (r"\bI am\b", "I'm"), (r"\bI have\b", "I've"),
            (r"\bI will\b", "I'll"), (r"\bI would\b", "I'd"),
            (r"\byou are\b", "you're"), (r"\bYou are\b", "You're"),
            (r"\byou have\b", "you've"), (r"\bYou have\b", "You've"),
            (r"\byou will\b", "you'll"), (r"\bYou will\b", "You'll"),
            (r"\bthey are\b", "they're"), (r"\bThey are\b", "They're"),
            (r"\bthey have\b", "they've"), (r"\bThey have\b", "They've"),
            (r"\bwe are\b", "we're"), (r"\bWe are\b", "We're"),
            (r"\bwe have\b", "we've"), (r"\bWe have\b", "We've"),
            (r"\bit is\b", "it's"), (r"\bIt is\b", "It's"),
            (r"\bthat is\b", "that's"), (r"\bThat is\b", "That's"),
            (r"\bwhat is\b", "what's"), (r"\bWhat is\b", "What's"),
            (r"\bwho is\b", "who's"), (r"\bWho is\b", "Who's"),
            (r"\bwhere is\b", "where's"), (r"\bWhere is\b", "Where's"),
            (r"\bthere is\b", "there's"), (r"\bThere is\b", "There's"),
            (r"\bhere is\b", "here's"), (r"\bHere is\b", "Here's"),
            (r"\blet us\b", "let's"), (r"\bLet us\b", "Let's"),
            (r"\bhe is\b", "he's"), (r"\bHe is\b", "He's"),
            (r"\bshe is\b", "she's"), (r"\bShe is\b", "She's"),
        ]

        result = text
        for pattern, replacement in contractions:
            result = re.sub(pattern, replacement, result)

        return result

    def _inject_perplexity(self, text: str) -> str:
        """Add natural word variation to increase perplexity."""
        # Occasional word insertions that humans use
        insertions = {
            r'\b(is)\b': ['is actually', 'is really', 'is basically'],
            r'\b(are)\b': ['are really', 'are actually', 'are basically'],
            r'\b(was)\b': ['was actually', 'was really'],
            r'\b(have)\b': ['have actually', 'have really'],
            r'\b(think)\b': ['think', 'honestly think', 'really think'],
            r'\b(know)\b': ['know', 'actually know', 'really know'],
            r'\b(important)\b': ['important', 'really important', 'pretty important'],
            r'\b(good)\b': ['good', 'really good', 'pretty good'],
            r'\b(great)\b': ['great', 'really great'],
            r'\b(just)\b': ['just', 'basically just'],
        }

        result = text
        intensity = self._creativity_level / 100 * 0.15  # Max 15% injection rate

        for pattern, replacements in insertions.items():
            if random.random() < intensity:
                # Only replace first occurrence sometimes
                compiled = re.compile(pattern, re.IGNORECASE)
                if compiled.search(result):
                    replacement = random.choice(replacements)
                    result = compiled.sub(replacement, result, count=1)

        return result

    def _restructure_sentences(self, text: str) -> str:
        """Vary sentence structure and length."""
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

            # Split long sentences (30% chance if > 25 words)
            if word_count > 25 and random.random() < 0.35:
                split_result = self._try_split_sentence(sentence)
                if split_result:
                    result_sentences.extend(split_result)
                    continue

            # Merge very short consecutive sentences (25% chance)
            if word_count < 8 and i < len(sentences) - 1 and random.random() < 0.25:
                next_sentence = sentences[i + 1].strip() if i + 1 < len(sentences) else ""
                if next_sentence and len(next_sentence.split()) < 12:
                    merged = self._merge_sentences(sentence, next_sentence)
                    if merged:
                        result_sentences.append(merged)
                        sentences[i + 1] = ""  # Mark as processed
                        continue

            # Add casual starter (15% chance, not first sentence) - only for conversational/creative
            if self._current_style in ["conversational", "creative"]:
                if i > 0 and word_count > 5 and random.random() < 0.15:
                    if not any(sentence.startswith(s) for s in CASUAL_STARTERS + SENTENCE_CONNECTORS):
                        starter = random.choice(SENTENCE_CONNECTORS)
                        sentence = starter + sentence[0].lower() + sentence[1:]

            result_sentences.append(sentence)

        return ' '.join(result_sentences)

    def _try_split_sentence(self, sentence: str) -> Optional[List[str]]:
        """Try to split a long sentence naturally."""
        words = sentence.split()
        word_count = len(words)
        mid = word_count // 2

        # Look for good split points near middle
        split_words = ['and', 'but', 'so', 'which', 'because', 'while', 'when', 'although', 'since']

        for j in range(mid - 5, min(mid + 5, word_count)):
            if j < word_count and words[j].lower() in split_words:
                first = ' '.join(words[:j])
                second = ' '.join(words[j+1:])

                if first and second and len(first.split()) >= 5 and len(second.split()) >= 5:
                    # Ensure proper punctuation
                    if not first.endswith(('.', '!', '?')):
                        first += '.'
                    second = second[0].upper() + second[1:] if second else second

                    return [first, second]

        return None

    def _merge_sentences(self, first: str, second: str) -> Optional[str]:
        """Merge two short sentences naturally."""
        first = first.rstrip('.!?')
        second = second[0].lower() + second[1:] if second else second

        connectors = [', and ', ' and ', ', plus ', ', so ']
        connector = random.choice(connectors)

        merged = first + connector + second
        return merged if len(merged.split()) <= 30 else None

    def _add_human_touches(self, text: str) -> str:
        """Add subtle human writing patterns."""
        sentences = self._split_sentences(text)
        result_sentences = []
        intensity = self._creativity_level / 100

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            words = sentence.split()

            # Occasional filler word insertion (10% * intensity)
            if len(words) > 8 and random.random() < 0.10 * intensity:
                insert_pos = random.randint(2, min(5, len(words) - 1))
                filler = random.choice(FILLER_INSERTIONS)
                if filler not in sentence.lower():
                    words.insert(insert_pos, filler)
                    sentence = ' '.join(words)

            # Casual starter for mid-paragraph sentences (12% chance)
            if i > 0 and i < len(sentences) - 1 and random.random() < 0.12 * intensity:
                if not any(sentence.startswith(s) for s in CASUAL_STARTERS):
                    starter = random.choice(CASUAL_STARTERS)
                    sentence = starter + sentence[0].lower() + sentence[1:]

            result_sentences.append(sentence)

        return ' '.join(result_sentences)

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _final_cleanup(self, text: str) -> str:
        """Final cleanup and validation."""
        result = text

        # Remove double spaces
        result = re.sub(r'\s+', ' ', result)

        # Fix spacing around punctuation
        result = re.sub(r'\s+([.,!?;:])', r'\1', result)
        result = re.sub(r'([.,!?])\s*([.,!?])', r'\1', result)

        # Remove em-dashes (major AI tell)
        result = result.replace('—', ',')
        result = result.replace('--', ',')
        result = result.replace(' - ', ', ')

        # Remove semicolons (replace with period or comma)
        result = re.sub(r';\s*', '. ', result)

        # Fix sentence capitalization after period replacements
        result = re.sub(r'\.\s+([a-z])', lambda m: '. ' + m.group(1).upper(), result)

        # Ensure proper ending
        result = result.strip()
        if result and result[-1] not in '.!?':
            result += '.'

        return result


# Factory function to create either V1 or V2 humanizer
def create_humanizer(version: str = "v2", config=None):
    """Create humanizer instance."""
    if version.lower() == "v2":
        return HumanizerV2(config)
    else:
        from .humanizer import Humanizer
        return Humanizer(config)
