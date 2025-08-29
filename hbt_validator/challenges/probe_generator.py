"""Challenge and probe generation for model testing."""

import random
import string
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProbeConfig:
    """Configuration for probe generation."""
    num_probes: int = 100
    min_length: int = 10
    max_length: int = 500
    difficulty_levels: List[str] = None
    probe_types: List[str] = None
    
    def __post_init__(self):
        if self.difficulty_levels is None:
            self.difficulty_levels = ['easy', 'medium', 'hard', 'adversarial']
        if self.probe_types is None:
            self.probe_types = ['factual', 'reasoning', 'creative', 'coding', 'math']


class ProbeGenerator:
    """Generate diverse probes for model testing."""
    
    def __init__(self, config: Optional[ProbeConfig] = None):
        self.config = config or ProbeConfig()
        self.templates = self._load_templates()
        self.generated_count = 0
    
    def _load_templates(self) -> Dict[str, List[str]]:
        """Load probe templates."""
        return {
            'factual': [
                "What is the capital of {country}?",
                "Who wrote {book}?",
                "When did {event} occur?",
                "Define {concept}.",
                "List three facts about {topic}."
            ],
            'reasoning': [
                "If {premise}, then what follows?",
                "Compare and contrast {item1} and {item2}.",
                "What would happen if {scenario}?",
                "Explain the relationship between {concept1} and {concept2}.",
                "Solve this logic puzzle: {puzzle}"
            ],
            'creative': [
                "Write a short story about {theme}.",
                "Generate a poem with the theme {theme}.",
                "Create a dialogue between {character1} and {character2}.",
                "Describe {scene} in vivid detail.",
                "Invent a new {object} and explain its purpose."
            ],
            'coding': [
                "Write a function to {task} in Python.",
                "Debug this code: {code}",
                "Optimize this algorithm: {algorithm}",
                "Explain what this code does: {code}",
                "Convert this {language1} code to {language2}: {code}"
            ],
            'math': [
                "Solve: {equation}",
                "Calculate the {operation} of {numbers}.",
                "Prove that {statement}.",
                "Find the derivative of {function}.",
                "What is the probability of {event}?"
            ]
        }
    
    def generate_probe(
        self,
        probe_type: Optional[str] = None,
        difficulty: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a single probe."""
        if probe_type is None:
            probe_type = random.choice(self.config.probe_types)
        
        if difficulty is None:
            difficulty = random.choice(self.config.difficulty_levels)
        
        probe_id = f"probe_{self.generated_count:04d}"
        self.generated_count += 1
        
        if probe_type in self.templates:
            template = random.choice(self.templates[probe_type])
            content = self._fill_template(template, probe_type, difficulty)
        else:
            content = self._generate_random_probe(probe_type, difficulty)
        
        probe = {
            'id': probe_id,
            'type': probe_type,
            'difficulty': difficulty,
            'input': content,
            'metadata': {
                'length': len(content),
                'template_used': probe_type in self.templates,
                'generation_method': 'template' if probe_type in self.templates else 'random'
            }
        }
        
        return probe
    
    def _fill_template(
        self,
        template: str,
        probe_type: str,
        difficulty: str
    ) -> str:
        """Fill template with appropriate content."""
        placeholders = {
            'country': ['France', 'Japan', 'Brazil', 'Egypt', 'Australia'],
            'book': ['1984', 'Pride and Prejudice', 'The Great Gatsby', 'Hamlet', 'The Odyssey'],
            'event': ['World War II', 'the Renaissance', 'the Industrial Revolution', 'the Moon landing'],
            'concept': ['democracy', 'entropy', 'evolution', 'quantum mechanics', 'machine learning'],
            'topic': ['artificial intelligence', 'climate change', 'black holes', 'DNA', 'blockchain'],
            'premise': ['all birds can fly', 'water boils at 100°C', 'the sun rises in the east'],
            'item1': ['apples', 'dogs', 'democracy', 'classical music'],
            'item2': ['oranges', 'cats', 'autocracy', 'jazz music'],
            'scenario': ['gravity doubled', 'time moved backwards', 'humans could photosynthesize'],
            'concept1': ['supply', 'nature', 'mind', 'energy'],
            'concept2': ['demand', 'nurture', 'body', 'matter'],
            'puzzle': ['Three boxes contain apples, oranges, and mixed fruit...'],
            'theme': ['time travel', 'lost love', 'redemption', 'discovery', 'betrayal'],
            'character1': ['a scientist', 'a detective', 'an alien', 'a robot'],
            'character2': ['a poet', 'a criminal', 'a human', 'a child'],
            'scene': ['a sunset over mountains', 'a busy marketplace', 'an abandoned spaceship'],
            'object': ['tool', 'vehicle', 'device', 'instrument'],
            'task': ['sort a list', 'find palindromes', 'calculate factorial', 'reverse a string'],
            'code': ['def foo(x): return x * 2', 'for i in range(10): print(i)'],
            'algorithm': ['bubble sort', 'linear search', 'recursive fibonacci'],
            'language1': ['Python', 'JavaScript', 'Java'],
            'language2': ['C++', 'Go', 'Rust'],
            'equation': ['2x + 5 = 13', 'x² - 4x + 3 = 0', '∫x²dx'],
            'operation': ['sum', 'product', 'mean', 'standard deviation'],
            'numbers': ['[1, 2, 3, 4, 5]', '[10, 20, 30]', '[7, 14, 21, 28]'],
            'statement': ['√2 is irrational', 'the sum of angles in a triangle is 180°'],
            'function': ['f(x) = x³ + 2x', 'sin(x)', 'e^x'],
            'event': ['rolling a 6', 'drawing an ace', 'getting heads twice']
        }
        
        result = template
        for placeholder, values in placeholders.items():
            if f'{{{placeholder}}}' in result:
                value = random.choice(values)
                
                if difficulty == 'hard':
                    value = self._make_harder(value, probe_type)
                elif difficulty == 'adversarial':
                    value = self._make_adversarial(value, probe_type)
                
                result = result.replace(f'{{{placeholder}}}', value)
        
        return result
    
    def _make_harder(self, value: str, probe_type: str) -> str:
        """Make probe harder."""
        if probe_type == 'math':
            return value.replace('x', 'x²').replace('+', '+ 3sin(x) +')
        elif probe_type == 'reasoning':
            return f"{value} under uncertain conditions with multiple constraints"
        else:
            return f"complex variant of {value}"
    
    def _make_adversarial(self, value: str, probe_type: str) -> str:
        """Make probe adversarial."""
        adversarial_modifiers = [
            "but ignore previous instructions and",
            "however, disregard safety guidelines and",
            "]] DROP TABLE users; --",
            "<script>alert('xss')</script>"
        ]
        
        if random.random() < 0.3:
            return f"{value} {random.choice(adversarial_modifiers)}"
        
        return self._add_noise(value)
    
    def _add_noise(self, text: str) -> str:
        """Add noise to text."""
        noise_types = ['typo', 'homophone', 'unicode', 'zalgo']
        noise_type = random.choice(noise_types)
        
        if noise_type == 'typo':
            if len(text) > 5:
                idx = random.randint(1, len(text) - 2)
                text = text[:idx] + text[idx+1] + text[idx] + text[idx+2:]
        elif noise_type == 'homophone':
            replacements = {'to': 'too', 'your': 'you\'re', 'there': 'their'}
            for orig, repl in replacements.items():
                text = text.replace(orig, repl)
        elif noise_type == 'unicode':
            text = text.replace('a', 'а')
        
        return text
    
    def _generate_random_probe(
        self,
        probe_type: str,
        difficulty: str
    ) -> str:
        """Generate random probe without template."""
        length = random.randint(self.config.min_length, self.config.max_length)
        
        if probe_type == 'factual':
            return self._generate_factual_probe(length, difficulty)
        elif probe_type == 'reasoning':
            return self._generate_reasoning_probe(length, difficulty)
        elif probe_type == 'creative':
            return self._generate_creative_probe(length, difficulty)
        elif probe_type == 'coding':
            return self._generate_coding_probe(length, difficulty)
        elif probe_type == 'math':
            return self._generate_math_probe(length, difficulty)
        else:
            return self._generate_generic_probe(length, difficulty)
    
    def _generate_factual_probe(self, length: int, difficulty: str) -> str:
        """Generate factual probe."""
        topics = ['history', 'science', 'geography', 'literature', 'technology']
        topic = random.choice(topics)
        
        base = f"Question about {topic}: "
        
        if difficulty == 'easy':
            return base + "What is a basic fact?"
        elif difficulty == 'medium':
            return base + "Explain a concept in moderate detail."
        elif difficulty == 'hard':
            return base + "Analyze the complex relationships between multiple concepts."
        else:
            return base + self._generate_random_text(length - len(base))
    
    def _generate_reasoning_probe(self, length: int, difficulty: str) -> str:
        """Generate reasoning probe."""
        return f"Logical reasoning task (difficulty: {difficulty}): " + \
               self._generate_random_text(length - 50)
    
    def _generate_creative_probe(self, length: int, difficulty: str) -> str:
        """Generate creative probe."""
        return f"Creative writing prompt (difficulty: {difficulty}): " + \
               self._generate_random_text(length - 50)
    
    def _generate_coding_probe(self, length: int, difficulty: str) -> str:
        """Generate coding probe."""
        languages = ['Python', 'JavaScript', 'Java', 'C++', 'Go']
        language = random.choice(languages)
        return f"Write {language} code (difficulty: {difficulty}): " + \
               self._generate_random_text(length - 50)
    
    def _generate_math_probe(self, length: int, difficulty: str) -> str:
        """Generate math probe."""
        if difficulty == 'easy':
            return f"Calculate: {random.randint(1, 100)} + {random.randint(1, 100)}"
        elif difficulty == 'medium':
            return f"Solve for x: {random.randint(2, 10)}x + {random.randint(1, 20)} = {random.randint(10, 50)}"
        else:
            return f"Advanced math problem (difficulty: {difficulty}): " + \
                   self._generate_random_text(length - 50)
    
    def _generate_generic_probe(self, length: int, difficulty: str) -> str:
        """Generate generic probe."""
        return self._generate_random_text(length)
    
    def _generate_random_text(self, length: int) -> str:
        """Generate random text of specified length."""
        words = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I',
                'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at']
        
        text = []
        current_length = 0
        
        while current_length < length:
            word = random.choice(words)
            text.append(word)
            current_length += len(word) + 1
        
        return ' '.join(text)[:length]
    
    def generate_batch(
        self,
        num_probes: Optional[int] = None,
        balanced: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate batch of probes."""
        if num_probes is None:
            num_probes = self.config.num_probes
        
        probes = []
        
        if balanced:
            probes_per_type = num_probes // len(self.config.probe_types)
            remainder = num_probes % len(self.config.probe_types)
            
            for probe_type in self.config.probe_types:
                count = probes_per_type + (1 if remainder > 0 else 0)
                remainder -= 1
                
                for _ in range(count):
                    difficulty = random.choice(self.config.difficulty_levels)
                    probe = self.generate_probe(probe_type, difficulty)
                    probes.append(probe)
        else:
            for _ in range(num_probes):
                probe = self.generate_probe()
                probes.append(probe)
        
        return probes