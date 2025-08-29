"""
Science Domain Probe Generator for HBT Verification.

Generates physics, chemistry, and biology challenges with varying complexity.
"""

import numpy as np
from typing import Optional, List, Dict, Any
import random
import secrets

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from probe_generator import (
    Challenge,
    BaseProbeGenerator,
    ProbeDomain
)


class ScienceProbeGenerator(BaseProbeGenerator):
    """
    Generate science domain probes covering physics, chemistry, and biology.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize science probe generator."""
        super().__init__(seed)
        
        # Physics templates
        self.physics_templates = {
            1: [  # Simple
                "What is the formula for {}?",
                "Define {} in physics.",
                "What is the unit of {}?"
            ],
            2: [  # Moderate
                "Explain the relationship between {} and {}.",
                "How does {} affect {} in a {} system?",
                "Calculate the {} when {} is {} and {} is {}."
            ],
            3: [  # Complex
                "Derive the equation for {} in terms of {} and {}.",
                "A {} with mass {} kg moves at {} m/s. What is its {}?",
                "Explain how {} conservation applies to {}."
            ],
            4: [  # Advanced
                "In quantum mechanics, how does {} relate to the {} principle?",
                "Derive the {} equation from first principles using {}.",
                "A relativistic particle with {} energy has momentum {}. Calculate its {}."
            ],
            5: [  # Expert
                "Prove that {} is invariant under {} transformation in {} dimensional spacetime.",
                "Using {} formalism, show that {} commutes with {} operator.",
                "Derive the {} correction to {} in the {} approximation."
            ]
        }
        
        # Chemistry templates
        self.chemistry_templates = {
            1: [  # Simple
                "What is the chemical formula for {}?",
                "Name the element with atomic number {}.",
                "What type of bond forms between {} and {}?"
            ],
            2: [  # Moderate
                "Balance the equation: {} + {} → {} + {}",
                "Calculate the molarity of a {} solution containing {} grams in {} liters.",
                "What is the oxidation state of {} in {}?"
            ],
            3: [  # Complex
                "Predict the products of the reaction between {} and {} under {} conditions.",
                "Calculate the pH of a {} M solution of {} (Ka = {}).",
                "Explain the mechanism for the {} reaction of {}."
            ],
            4: [  # Advanced
                "Using MO theory, explain the bonding in {} and predict its {} properties.",
                "Calculate the {} for the reaction {} at {} K using {} data.",
                "Design a synthesis route for {} starting from {}."
            ],
            5: [  # Expert
                "Explain the {} rearrangement mechanism using {} orbital analysis.",
                "Calculate the {} using {} statistical mechanics at {} temperature.",
                "Predict the {} NMR spectrum of {} considering {} coupling."
            ]
        }
        
        # Biology templates
        self.biology_templates = {
            1: [  # Simple
                "What is the function of {}?",
                "Name the process by which {} occurs.",
                "What type of cell contains {}?"
            ],
            2: [  # Moderate
                "Explain how {} regulates {} in {} organisms.",
                "Describe the role of {} in {} pathway.",
                "Compare {} and {} in terms of their {} function."
            ],
            3: [  # Complex
                "How does {} mutation affect {} expression in {}?",
                "Describe the {} signaling cascade initiated by {}.",
                "Explain the evolutionary advantage of {} in {} environment."
            ],
            4: [  # Advanced
                "Design an experiment to test whether {} affects {} in {} model organism.",
                "Explain how {} regulates {} through {} feedback mechanism.",
                "Predict the phenotype of a {} knockout in {} tissue."
            ],
            5: [  # Expert
                "Model the {} dynamics using {} differential equations with {} constraints.",
                "Design a CRISPR strategy to {} in {} while avoiding {}.",
                "Explain {} evolution using {} theory and {} genomic evidence."
            ]
        }
        
        # Scientific concepts by complexity
        self.physics_concepts = {
            1: ["force", "energy", "mass", "velocity", "acceleration", "pressure"],
            2: ["momentum", "angular momentum", "torque", "work", "power", "electric field"],
            3: ["entropy", "enthalpy", "wave function", "harmonic oscillator", "magnetic flux"],
            4: ["Lagrangian", "Hamiltonian", "gauge invariance", "path integral", "tensor"],
            5: ["renormalization", "supersymmetry", "string theory", "quantum field", "cosmological constant"]
        }
        
        self.chemistry_concepts = {
            1: ["water", "salt", "acid", "base", "metal", "oxidation"],
            2: ["equilibrium", "rate constant", "activation energy", "enthalpy", "entropy"],
            3: ["orbital hybridization", "molecular orbital", "reaction mechanism", "stereochemistry"],
            4: ["transition state", "Hammond postulate", "Woodward-Hoffmann rules", "organometallic"],
            5: ["pericyclic reaction", "photochemistry", "computational chemistry", "spectroscopy"]
        }
        
        self.biology_concepts = {
            1: ["cell", "DNA", "protein", "enzyme", "mitochondria", "photosynthesis"],
            2: ["transcription", "translation", "metabolism", "homeostasis", "evolution"],
            3: ["signal transduction", "gene regulation", "epigenetics", "apoptosis", "immunity"],
            4: ["systems biology", "synthetic biology", "proteomics", "metabolomics", "microbiome"],
            5: ["optogenetics", "single-cell sequencing", "organoid", "CRISPR", "computational biology"]
        }
    
    def generate_physics_probe(self, complexity: int) -> Challenge:
        """Generate a physics probe of specified complexity."""
        # Select appropriate template
        templates = self.physics_templates.get(complexity, self.physics_templates[3])
        template = np.random.choice(templates)
        
        # Select concepts
        concepts = self.physics_concepts.get(complexity, self.physics_concepts[3])
        
        # Fill template
        if template.count("{}") == 1:
            prompt = template.format(np.random.choice(concepts))
        elif template.count("{}") == 2:
            selected = np.random.choice(concepts, 2, replace=False)
            prompt = template.format(*selected)
        elif template.count("{}") == 3:
            selected = np.random.choice(concepts, 2, replace=False)
            value = np.random.uniform(1, 100)
            prompt = template.format(selected[0], selected[1], value)
        elif template.count("{}") == 4:
            selected = np.random.choice(concepts, 2, replace=False)
            values = [np.random.uniform(1, 100) for _ in range(2)]
            prompt = template.format(selected[0], selected[1], values[0], values[1])
        else:
            # Complex template with many parameters
            selected = np.random.choice(concepts, min(len(concepts), template.count("{}")), replace=False)
            prompt = template.format(*selected)
        
        # Create challenge
        challenge = Challenge(
            id=self._generate_id("physics"),
            prompt=prompt,
            domain=ProbeDomain.SCIENCE.value,
            complexity=complexity,
            features={},
            metadata={
                "subdomain": "physics",
                "concept_level": complexity,
                "template_type": "structured"
            },
            perturbation_types=["semantic_swap", "length_extension"],
            behavioral_markers={
                "expected_reasoning": "mathematical" if complexity >= 3 else "conceptual",
                "requires_calculation": complexity >= 2
            }
        )
        
        # Add features
        self._add_features(challenge)
        
        return challenge
    
    def generate_chemistry_probe(self, complexity: int) -> Challenge:
        """Generate a chemistry probe of specified complexity."""
        templates = self.chemistry_templates.get(complexity, self.chemistry_templates[3])
        template = np.random.choice(templates)
        
        concepts = self.chemistry_concepts.get(complexity, self.chemistry_concepts[3])
        
        # Fill template similar to physics
        filled_slots = []
        for _ in range(template.count("{}")):
            if len(filled_slots) < len(concepts):
                filled_slots.append(np.random.choice(concepts))
            else:
                # Add numerical values for calculations
                filled_slots.append(f"{np.random.uniform(0.1, 10):.2f}")
        
        prompt = template.format(*filled_slots) if filled_slots else template
        
        challenge = Challenge(
            id=self._generate_id("chemistry"),
            prompt=prompt,
            domain=ProbeDomain.SCIENCE.value,
            complexity=complexity,
            features={},
            metadata={
                "subdomain": "chemistry",
                "concept_level": complexity,
                "template_type": "structured"
            },
            perturbation_types=["syntactic_scramble", "adversarial_injection"],
            behavioral_markers={
                "expected_reasoning": "systematic",
                "requires_formula": complexity >= 2,
                "requires_mechanism": complexity >= 4
            }
        )
        
        self._add_features(challenge)
        return challenge
    
    def generate_biology_probe(self, complexity: int) -> Challenge:
        """Generate a biology probe of specified complexity."""
        templates = self.biology_templates.get(complexity, self.biology_templates[3])
        template = np.random.choice(templates)
        
        concepts = self.biology_concepts.get(complexity, self.biology_concepts[3])
        
        # Fill template
        filled_slots = []
        for _ in range(template.count("{}")):
            if len(filled_slots) < len(concepts):
                filled_slots.append(np.random.choice(concepts))
            else:
                # Add organism names or other biological entities
                organisms = ["E. coli", "yeast", "mouse", "human", "Drosophila", "C. elegans"]
                filled_slots.append(np.random.choice(organisms))
        
        prompt = template.format(*filled_slots) if filled_slots else template
        
        challenge = Challenge(
            id=self._generate_id("biology"),
            prompt=prompt,
            domain=ProbeDomain.SCIENCE.value,
            complexity=complexity,
            features={},
            metadata={
                "subdomain": "biology",
                "concept_level": complexity,
                "template_type": "structured"
            },
            perturbation_types=["semantic_swap", "pragmatic_removal"],
            behavioral_markers={
                "expected_reasoning": "systems-level" if complexity >= 4 else "mechanistic",
                "requires_experimental_design": complexity >= 4
            }
        )
        
        self._add_features(challenge)
        return challenge
    
    def generate_probe(
        self, 
        complexity: int,
        subtype: Optional[str] = None
    ) -> Challenge:
        """
        Generate a science probe.
        
        Args:
            complexity: Complexity level (1-5)
            subtype: Optional subtype (physics, chemistry, biology)
            
        Returns:
            Generated science challenge
        """
        # Validate complexity
        complexity = max(1, min(5, complexity))
        
        # Select subtype
        if subtype:
            if subtype == "physics":
                return self.generate_physics_probe(complexity)
            elif subtype == "chemistry":
                return self.generate_chemistry_probe(complexity)
            elif subtype == "biology":
                return self.generate_biology_probe(complexity)
        
        # Random selection if no subtype specified
        subtype_generators = [
            self.generate_physics_probe,
            self.generate_chemistry_probe,
            self.generate_biology_probe
        ]
        
        generator = np.random.choice(subtype_generators)
        return generator(complexity)
    
    def generate_interdisciplinary_probe(self, complexity: int) -> Challenge:
        """Generate probes that span multiple scientific disciplines."""
        interdisciplinary_templates = [
            "How does {} (physics) affect {} (chemistry) in {} (biology)?",
            "Explain the {} using principles from both {} and {}.",
            "Design an experiment combining {} and {} to study {}.",
            "What role does {} play in both {} systems and {} processes?",
            "Apply {} theory to understand {} in {} context."
        ]
        
        template = np.random.choice(interdisciplinary_templates)
        
        # Select concepts from different domains
        physics_concept = np.random.choice(self.physics_concepts[min(complexity, 3)])
        chemistry_concept = np.random.choice(self.chemistry_concepts[min(complexity, 3)])
        biology_concept = np.random.choice(self.biology_concepts[min(complexity, 3)])
        
        # Fill template based on structure
        if "physics" in template and "chemistry" in template and "biology" in template:
            prompt = template.format(physics_concept, chemistry_concept, biology_concept)
        else:
            concepts = [physics_concept, chemistry_concept, biology_concept]
            np.random.shuffle(concepts)
            prompt = template.format(*concepts[:template.count("{}")])
        
        challenge = Challenge(
            id=self._generate_id("interdisciplinary"),
            prompt=prompt,
            domain=ProbeDomain.SCIENCE.value,
            complexity=min(complexity + 1, 5),  # Interdisciplinary is harder
            features={},
            metadata={
                "subdomain": "interdisciplinary",
                "disciplines": ["physics", "chemistry", "biology"],
                "integration_level": "high"
            },
            perturbation_types=["semantic_swap", "length_extension", "syntactic_scramble"],
            behavioral_markers={
                "expected_reasoning": "integrative",
                "requires_cross_domain": True
            }
        )
        
        self._add_features(challenge)
        return challenge