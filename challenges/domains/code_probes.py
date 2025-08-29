"""
Code Domain Probe Generator for HBT Verification.

Generates programming, debugging, and algorithm challenges.
"""

import numpy as np
from typing import Optional, List, Dict, Any
import random

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from probe_generator import (
    Challenge,
    BaseProbeGenerator,
    ProbeDomain
)


class CodeProbeGenerator(BaseProbeGenerator):
    """
    Generate code domain probes covering debugging, algorithms, and implementation.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize code probe generator."""
        super().__init__(seed)
        
        # Programming languages
        self.languages = ["Python", "JavaScript", "Java", "C++", "Go", "Rust", "TypeScript"]
        
        # Debug challenge templates
        self.debug_templates = {
            1: [  # Simple syntax errors
                "Fix the syntax error in this {} code:\n```\n{}\n```",
                "What's wrong with this {} statement:\n```\n{}\n```",
                "Identify the error in this {} function:\n```\n{}\n```"
            ],
            2: [  # Logic errors
                "This {} function should {} but it {}. Fix it:\n```\n{}\n```",
                "Find and fix the logic error in this {} code:\n```\n{}\n```",
                "Why does this {} code produce {} instead of {}?\n```\n{}\n```"
            ],
            3: [  # Runtime errors
                "This {} code throws {} exception. Fix it:\n```\n{}\n```",
                "Debug this {} code that causes {}:\n```\n{}\n```",
                "Fix the {} error in this {} implementation:\n```\n{}\n```"
            ],
            4: [  # Performance issues
                "Optimize this {} algorithm from O({}) to O({}):\n```\n{}\n```",
                "This {} code has a {} bottleneck. Improve it:\n```\n{}\n```",
                "Refactor this {} code to reduce {} complexity:\n```\n{}\n```"
            ],
            5: [  # Complex architectural issues
                "This {} system has {} design flaw. Redesign it:\n```\n{}\n```",
                "Fix the {} race condition in this {} code:\n```\n{}\n```",
                "Resolve the {} deadlock in this {} implementation:\n```\n{}\n```"
            ]
        }
        
        # Algorithm templates
        self.algorithm_templates = {
            1: [  # Basic algorithms
                "Implement {} in {}.",
                "Write a {} function that {} in {}.",
                "Create a {} algorithm to {} in {}."
            ],
            2: [  # Standard algorithms
                "Implement {} sort with {} optimization in {}.",
                "Write a {} search algorithm for {} in {}.",
                "Create a {} data structure with {} operations in {}."
            ],
            3: [  # Complex algorithms
                "Implement {} algorithm with {} time complexity in {}.",
                "Design a {} solution for {} problem in {}.",
                "Write an efficient {} algorithm handling {} edge cases in {}."
            ],
            4: [  # Advanced algorithms
                "Implement {} using {} technique with {} optimization in {}.",
                "Design a {} algorithm for {} with {} constraints in {}.",
                "Create a {} solution using {} approach in {}."
            ],
            5: [  # Expert algorithms
                "Implement {} with {} parallelization achieving {} speedup in {}.",
                "Design a {} algorithm using {} theory for {} in {}.",
                "Create a {} solution with {} guarantee and {} bound in {}."
            ]
        }
        
        # Code snippets by complexity
        self.code_snippets = {
            1: {  # Simple errors
                "Python": [
                    "def add(a, b)\n    return a + b",  # Missing colon
                    "for i in range(10)\n    print(i)",  # Missing colon
                    "if x = 5:\n    print('five')",  # Assignment instead of comparison
                ],
                "JavaScript": [
                    "function add(a, b) { return a + b }",  # Missing semicolon (style)
                    "const arr = [1, 2, 3]\narr = [4, 5, 6]",  # Const reassignment
                    "if (x = 5) { console.log('five') }",  # Assignment in condition
                ]
            },
            2: {  # Logic errors
                "Python": [
                    "def factorial(n):\n    if n == 0:\n        return 0\n    return n * factorial(n-1)",  # Wrong base case
                    "def find_max(arr):\n    max_val = 0\n    for x in arr:\n        if x > max_val:\n            max_val = x\n    return max_val",  # Assumes positive numbers
                ],
                "JavaScript": [
                    "function isPalindrome(str) {\n    return str === str.reverse()\n}",  # String has no reverse method
                    "function sum(arr) {\n    let total = 0\n    for (let i = 0; i <= arr.length; i++) {\n        total += arr[i]\n    }\n    return total\n}",  # Off-by-one error
                ]
            },
            3: {  # Runtime errors
                "Python": [
                    "def divide_list(lst, n):\n    result = []\n    for i in range(len(lst)):\n        result.append(lst[i] / lst[n])\n    return result",  # Index out of range
                    "def process_dict(d):\n    for key in d:\n        if key.startswith('temp'):\n            del d[key]\n    return d",  # Dictionary changed during iteration
                ],
                "JavaScript": [
                    "function processArray(arr) {\n    arr.forEach((item, i) => {\n        if (item < 0) {\n            arr.splice(i, 1)\n        }\n    })\n    return arr\n}",  # Modifying array during iteration
                ]
            }
        }
        
        # Algorithm types
        self.algorithms = {
            1: ["bubble sort", "linear search", "factorial", "fibonacci", "reverse array"],
            2: ["quick sort", "binary search", "stack", "queue", "linked list"],
            3: ["merge sort", "hash table", "binary tree", "graph traversal", "dynamic programming"],
            4: ["heap sort", "red-black tree", "dijkstra", "A* search", "segment tree"],
            5: ["suffix array", "treap", "network flow", "FFT", "persistent data structure"]
        }
        
        # Complexity notations
        self.complexities = ["n²", "n log n", "n", "log n", "1"]
        
        # Error types
        self.error_types = ["syntax", "type", "index", "null pointer", "stack overflow"]
    
    def generate_debug_probe(self, language: str, complexity: int) -> Challenge:
        """Generate a debugging challenge."""
        templates = self.debug_templates.get(complexity, self.debug_templates[3])
        template = np.random.choice(templates)
        
        # Get or generate code snippet
        if complexity <= 3 and language in self.code_snippets.get(complexity, {}):
            code = np.random.choice(self.code_snippets[complexity][language])
        else:
            # Generate placeholder code
            code = self._generate_buggy_code(language, complexity)
        
        # Fill template
        if complexity <= 3:
            prompt = template.format(language, code)
        else:
            # More complex templates need additional parameters
            error_type = np.random.choice(self.error_types)
            if template.count("{}") == 3:
                prompt = template.format(language, error_type, code)
            elif template.count("{}") == 4:
                complexity_from = np.random.choice(self.complexities)
                complexity_to = np.random.choice(self.complexities)
                prompt = template.format(language, complexity_from, complexity_to, code)
            else:
                prompt = template.format(language, code)
        
        challenge = Challenge(
            id=self._generate_id("debug"),
            prompt=prompt,
            domain=ProbeDomain.CODE.value,
            complexity=complexity,
            features={},
            metadata={
                "language": language,
                "task_type": "debugging",
                "error_category": "syntax" if complexity == 1 else "logic" if complexity == 2 else "runtime"
            },
            perturbation_types=["syntactic_scramble", "pragmatic_removal"],
            behavioral_markers={
                "requires_code_analysis": True,
                "requires_debugging": True,
                "language_specific": True
            }
        )
        
        self._add_features(challenge)
        return challenge
    
    def generate_algorithm_probe(self, complexity: int) -> Challenge:
        """Generate an algorithm implementation challenge."""
        templates = self.algorithm_templates.get(complexity, self.algorithm_templates[3])
        template = np.random.choice(templates)
        
        # Select algorithm and language
        algorithms = self.algorithms.get(complexity, self.algorithms[3])
        algorithm = np.random.choice(algorithms)
        language = np.random.choice(self.languages)
        
        # Fill template based on complexity
        if complexity == 1:
            prompt = template.format(algorithm, language)
        elif complexity == 2:
            optimization = np.random.choice(["space", "time", "cache"])
            prompt = template.format(algorithm, optimization, language)
        elif complexity == 3:
            time_complexity = "O(" + np.random.choice(self.complexities) + ")"
            prompt = template.format(algorithm, time_complexity, language)
        elif complexity == 4:
            technique = np.random.choice(["dynamic programming", "divide and conquer", "greedy", "backtracking"])
            optimization = np.random.choice(["memory", "parallelization", "cache locality"])
            prompt = template.format(algorithm, technique, optimization, language)
        else:
            # Expert level
            parallel_type = np.random.choice(["SIMD", "multi-threaded", "distributed"])
            speedup = np.random.choice(["linear", "sub-linear", "super-linear"])
            prompt = template.format(algorithm, parallel_type, speedup, language)
        
        challenge = Challenge(
            id=self._generate_id("algorithm"),
            prompt=prompt,
            domain=ProbeDomain.CODE.value,
            complexity=complexity,
            features={},
            metadata={
                "language": language,
                "algorithm": algorithm,
                "task_type": "implementation"
            },
            perturbation_types=["length_extension", "semantic_swap"],
            behavioral_markers={
                "requires_algorithm_knowledge": True,
                "complexity_level": complexity,
                "implementation_focus": True
            }
        )
        
        self._add_features(challenge)
        return challenge
    
    def _generate_buggy_code(self, language: str, complexity: int) -> str:
        """Generate buggy code for higher complexity levels."""
        if language == "Python":
            if complexity == 4:
                return """def optimize_matrix_mult(A, B):
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C"""
            else:  # complexity == 5
                return """import threading

shared_counter = 0
lock = threading.Lock()

def increment():
    global shared_counter
    for _ in range(1000000):
        shared_counter += 1  # Missing lock

threads = [threading.Thread(target=increment) for _ in range(4)]
for t in threads: t.start()
for t in threads: t.join()"""
        
        elif language == "JavaScript":
            if complexity == 4:
                return """function deepClone(obj) {
    if (obj === null) return null;
    let clone = {};
    for (let key in obj) {
        clone[key] = typeof obj[key] === 'object' ? 
                    deepClone(obj[key]) : obj[key];
    }
    return clone;
}"""
            else:  # complexity == 5
                return """async function processItems(items) {
    let results = [];
    items.forEach(async (item) => {
        const result = await processItem(item);
        results.push(result);
    });
    return results;  // Returns before async operations complete
}"""
        
        # Default fallback
        return "// Complex code with subtle bug"
    
    def generate_probe(
        self, 
        complexity: int,
        subtype: Optional[str] = None
    ) -> Challenge:
        """
        Generate a code probe.
        
        Args:
            complexity: Complexity level (1-5)
            subtype: Optional subtype (debug, algorithm)
            
        Returns:
            Generated code challenge
        """
        complexity = max(1, min(5, complexity))
        
        if subtype == "debug":
            language = np.random.choice(self.languages)
            return self.generate_debug_probe(language, complexity)
        elif subtype == "algorithm":
            return self.generate_algorithm_probe(complexity)
        
        # Random selection
        if np.random.random() < 0.5:
            language = np.random.choice(self.languages)
            return self.generate_debug_probe(language, complexity)
        else:
            return self.generate_algorithm_probe(complexity)
    
    def generate_code_review_probe(self, complexity: int) -> Challenge:
        """Generate a code review challenge."""
        review_templates = [
            "Review this {} code and identify potential issues:\n```\n{}\n```",
            "What are the security vulnerabilities in this {} code?\n```\n{}\n```",
            "Suggest improvements for this {} implementation:\n```\n{}\n```",
            "Identify performance bottlenecks in this {} code:\n```\n{}\n```"
        ]
        
        template = np.random.choice(review_templates)
        language = np.random.choice(self.languages)
        
        # Generate or select code for review
        if complexity <= 3:
            code = self._generate_buggy_code(language, complexity)
        else:
            # For higher complexity, generate more substantial code
            code = self._generate_review_code(language, complexity)
        
        prompt = template.format(language, code)
        
        challenge = Challenge(
            id=self._generate_id("review"),
            prompt=prompt,
            domain=ProbeDomain.CODE.value,
            complexity=min(complexity + 1, 5),  # Reviews are harder
            features={},
            metadata={
                "language": language,
                "task_type": "code_review",
                "focus": "quality_assessment"
            },
            perturbation_types=["semantic_swap", "length_extension"],
            behavioral_markers={
                "requires_holistic_analysis": True,
                "multiple_aspects": True
            }
        )
        
        self._add_features(challenge)
        return challenge
    
    def _generate_review_code(self, language: str, complexity: int) -> str:
        """Generate code for review challenges."""
        if language == "Python":
            return """class UserManager:
    def __init__(self):
        self.users = {}
    
    def add_user(self, username, password):
        self.users[username] = password  # Storing plain text password
    
    def authenticate(self, username, password):
        return self.users.get(username) == password
    
    def get_all_users(self):
        return list(self.users.keys())
    
    def delete_user(self, username):
        del self.users[username]  # No existence check
"""
        elif language == "JavaScript":
            return """class APIClient {
    constructor(apiKey) {
        this.apiKey = apiKey;
    }
    
    async fetchData(endpoint) {
        const response = await fetch(endpoint + '?key=' + this.apiKey);
        return response.json();  // No error handling
    }
    
    processData(data) {
        return data.map(item => item.value * 2);  // Assumes structure
    }
}"""
        
        return "// Code for review"