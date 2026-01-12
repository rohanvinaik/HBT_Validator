"""HBT - Holographic Behavioral Twin for LLM Verification."""

__version__ = "1.0.0"

# Core components
from core.hbt_constructor import HolographicBehavioralTwin, HBTConfig
from core.hdc_encoder import HyperdimensionalEncoder, HDCConfig
from core.variance_analyzer import VarianceAnalyzer, VarianceConfig
from core.rev_executor import REVExecutor, REVConfig

# Verification
from verification.fingerprint_matcher import (
    FingerprintMatcher,
    BehavioralFingerprint,
    FingerprintConfig,
    VerificationResult
)
from verification.zk_proofs import ZKProofSystem, ZKConfig

# Challenges
from challenges.probe_generator import ProbeGenerator, Challenge

__all__ = [
    "__version__",
    # Core
    "HolographicBehavioralTwin",
    "HBTConfig",
    "HyperdimensionalEncoder",
    "HDCConfig",
    "VarianceAnalyzer",
    "VarianceConfig",
    "REVExecutor",
    "REVConfig",
    # Verification
    "FingerprintMatcher",
    "BehavioralFingerprint",
    "FingerprintConfig",
    "VerificationResult",
    "ZKProofSystem",
    "ZKConfig",
    # Challenges
    "ProbeGenerator",
    "Challenge",
]
