"""
Natural language command parser for navigation commands.

Parses commands like:
  "Go to the blue vase"              → target="blue vase"
  "Find the red chair in the bedroom" → target="red chair", location="bedroom"
  "Navigate to the table near the sofa" → target="table", spatial_ref="sofa"
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedCommand:
    """Result of parsing a navigation command."""
    raw_text: str
    target_object: str           # the object to navigate to
    target_attribute: str = ""   # color/size/type modifier
    spatial_relation: str = ""   # "near", "next to", "in front of", etc.
    spatial_reference: str = ""  # reference object for spatial relation
    location: str = ""           # room/area constraint
    action: str = "navigate"     # navigate, find, explore

    @property
    def query_text(self) -> str:
        """Text to use for semantic map query or Grounding DINO prompt."""
        parts = []
        if self.target_attribute:
            parts.append(self.target_attribute)
        parts.append(self.target_object)
        return " ".join(parts)

    @property
    def dino_prompt(self) -> str:
        """Text prompt formatted for Grounding DINO."""
        return self.query_text + "."


class CommandParser:
    """
    Parse natural language navigation commands.

    Uses regex-based parsing (no ML dependency) with optional spaCy enhancement.
    """

    # Action keywords
    ACTION_PATTERNS = {
        "navigate": r"(?:go\s+to|navigate\s+to|move\s+to|head\s+to|drive\s+to|walk\s+to)",
        "find": r"(?:find|locate|search\s+for|look\s+for|where\s+is)",
        "explore": r"(?:explore|scan|survey|check)",
    }

    # Spatial relations
    SPATIAL_RELATIONS = [
        "next to", "near", "beside", "close to", "in front of",
        "behind", "to the left of", "to the right of", "on top of",
        "under", "above", "below", "between",
    ]

    # Location keywords (rooms)
    LOCATIONS = [
        "bedroom", "living room", "kitchen", "bathroom", "dining room",
        "hallway", "garage", "office", "study", "closet", "basement",
        "attic", "laundry room", "foyer", "entrance",
    ]

    # Color attributes
    COLORS = [
        "red", "blue", "green", "yellow", "white", "black", "brown",
        "orange", "purple", "pink", "gray", "grey", "beige",
    ]

    # Size attributes
    SIZES = ["big", "small", "large", "tiny", "tall", "short"]

    def __init__(self, use_spacy: bool = False):
        """
        Args:
            use_spacy: If True, load spaCy for enhanced NLP parsing.
                       Falls back to regex if spaCy not available.
        """
        self.nlp = None
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                print("Warning: spaCy not available, using regex-only parsing")

    def parse(self, command: str) -> ParsedCommand:
        """
        Parse a navigation command string.

        Args:
            command: Natural language command, e.g. "Go to the blue vase"

        Returns:
            ParsedCommand with extracted fields.
        """
        text = command.strip()
        text_lower = text.lower()

        # Detect action
        action = "navigate"
        for act, pattern in self.ACTION_PATTERNS.items():
            if re.search(pattern, text_lower):
                action = act
                break

        # Extract location constraint ("in the bedroom")
        location = ""
        for loc in self.LOCATIONS:
            pattern = rf"(?:in|at|from)\s+(?:the\s+)?{re.escape(loc)}"
            if re.search(pattern, text_lower):
                location = loc
                # Remove location phrase from further parsing
                text_lower = re.sub(pattern, "", text_lower)
                break

        # Extract spatial relation ("near the sofa", "next to the table")
        spatial_relation = ""
        spatial_reference = ""
        for rel in self.SPATIAL_RELATIONS:
            pattern = rf"{re.escape(rel)}\s+(?:the\s+)?(\w+(?:\s+\w+)?)"
            match = re.search(pattern, text_lower)
            if match:
                spatial_relation = rel
                spatial_reference = match.group(1).strip()
                text_lower = text_lower[:match.start()] + text_lower[match.end():]
                break

        # Remove action phrase to isolate the target
        for pattern in self.ACTION_PATTERNS.values():
            text_lower = re.sub(pattern, "", text_lower)

        # Clean up articles, prepositions
        text_lower = re.sub(r"\b(the|a|an|that|this|those)\b", "", text_lower)
        text_lower = re.sub(r"\s+", " ", text_lower).strip()
        text_lower = text_lower.strip(" .,!?")

        # Extract attribute (color, size)
        attribute = ""
        remaining = text_lower
        for color in self.COLORS:
            if re.search(rf"\b{color}\b", remaining):
                attribute = color
                remaining = re.sub(rf"\b{color}\b", "", remaining).strip()
                break
        if not attribute:
            for size in self.SIZES:
                if re.search(rf"\b{size}\b", remaining):
                    attribute = size
                    remaining = re.sub(rf"\b{size}\b", "", remaining).strip()
                    break

        # The remaining text is the target object
        target = remaining.strip()
        if not target:
            # Fallback: use the attribute as part of the target
            target = attribute
            attribute = ""

        return ParsedCommand(
            raw_text=command,
            target_object=target,
            target_attribute=attribute,
            spatial_relation=spatial_relation,
            spatial_reference=spatial_reference,
            location=location,
            action=action,
        )


# ---- Standalone test ----
if __name__ == "__main__":
    parser = CommandParser()

    test_commands = [
        "Go to the blue vase",
        "Find the red chair in the bedroom",
        "Navigate to the table near the sofa",
        "Look for the large lamp next to the desk",
        "Go to the kitchen",
        "Find the green plant",
        "Move to the chair in the living room",
        "Where is the white cabinet",
        "Go to the shelf behind the sofa",
        "Navigate to the bed in the bedroom",
        "Explore the hallway",
        "Find the vase on top of the table",
    ]

    for cmd in test_commands:
        result = parser.parse(cmd)
        print(f"Command: \"{cmd}\"")
        print(f"  Action: {result.action}")
        print(f"  Target: \"{result.target_object}\"")
        print(f"  Attribute: \"{result.target_attribute}\"")
        print(f"  Query: \"{result.query_text}\"")
        if result.spatial_relation:
            print(f"  Spatial: {result.spatial_relation} → {result.spatial_reference}")
        if result.location:
            print(f"  Location: {result.location}")
        print()
