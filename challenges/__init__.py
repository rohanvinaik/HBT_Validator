"""
HBT Challenge and Probe Generation System.

This module provides comprehensive probe generation for Holographic Behavioral Twin verification.
"""

from .probe_generator import (
    Challenge,
    ProbeDomain,
    ProbeFeatureExtractor,
    CryptographicCommitment,
    BaseProbeGenerator,
    AdaptiveProbeSelector,
    ProbeGenerator,
    create_default_probe_generator,
    generate_probe_set
)

from .domains import (
    ScienceProbeGenerator,
    CodeProbeGenerator
)

__all__ = [
    # Core classes
    'Challenge',
    'ProbeDomain',
    'ProbeFeatureExtractor',
    'CryptographicCommitment',
    'BaseProbeGenerator',
    'AdaptiveProbeSelector',
    'ProbeGenerator',
    
    # Domain generators
    'ScienceProbeGenerator',
    'CodeProbeGenerator',
    
    # Convenience functions
    'create_default_probe_generator',
    'generate_probe_set'
]