"""HBT Validator - Hypervector Behavioral Tree Validator for LLM Verification."""

__version__ = "0.1.0"

# Core components
from hbt_validator.core.hbt_constructor import HBTConstructor, HBTConfig
from hbt_validator.core.hdc_encoder import HyperdimensionalEncoder, HDCConfig
from hbt_validator.core.variance_analyzer import VarianceAnalyzer, VarianceConfig
from hbt_validator.core.rev_executor import REVExecutor, SegmentConfig
from hbt_validator.core.rev_executor_enhanced import (
    REVExecutor as REVExecutorEnhanced,
    REVConfig,
    SegmentSignature,
    MerkleNode
)

# Utils
from hbt_validator.utils.perturbations import (
    PerturbationOperator,
    TokenPerturbation,
    EmbeddingPerturbation,
    StructuralPerturbation,
    SemanticPerturbation,
    AdversarialPerturbation,
    CompositePerturbation
)
from hbt_validator.utils.hypervector_ops import (
    HypervectorOperations,
    SimilarityMetrics,
    HypervectorMemory,
    HypervectorEncoder as HVEncoder,
    HypervectorClustering,
    HypervectorCompression
)
from hbt_validator.utils.cryptography import (
    MerkleTree,
    CommitmentScheme,
    HashChain,
    PedersenCommitment,
    SignatureScheme,
    TimestampCommitment
)
from hbt_validator.utils.api_wrappers import (
    ModelAPIFactory,
    OpenAIWrapper,
    AnthropicWrapper,
    HuggingFaceWrapper,
    LocalModelWrapper,
    BatchAPIClient
)

# Verification
from hbt_validator.verification.fingerprint_matcher import (
    FingerprintMatcher,
    BehavioralFingerprint,
    IncrementalFingerprint,
    FingerprintConfig
)
from hbt_validator.verification.structural_inference import (
    CausalGraphRecovery,
    StructuralSimilarity,
    HierarchicalStructure,
    GraphConfig
)
from hbt_validator.verification.zk_proofs import (
    ZKProofSystem,
    SchnorrProof,
    RangeProof,
    VectorDistanceProof,
    MembershipProof,
    SigmaProtocol,
    ZKConfig
)

# Challenges
from hbt_validator.challenges.probe_generator import ProbeGenerator, ProbeConfig
from hbt_validator.challenges.datasets import (
    ProbeDataset,
    StandardProbeDataset,
    AdversarialProbeDataset,
    DomainSpecificDataset,
    DatasetManager
)

# Experiments
from hbt_validator.experiments.validation import ValidationExperiment, BatchValidation
from hbt_validator.experiments.ablations import AblationStudy
from hbt_validator.experiments.benchmarks import PerformanceBenchmark

__all__ = [
    # Version
    "__version__",
    
    # Core
    "HBTConstructor",
    "HBTConfig",
    "HyperdimensionalEncoder",
    "HDCConfig",
    "VarianceAnalyzer",
    "VarianceConfig",
    "REVExecutor",
    "REVExecutorEnhanced",
    "REVConfig",
    "SegmentConfig",
    "SegmentSignature",
    "MerkleNode",
    
    # Utils - Perturbations
    "PerturbationOperator",
    "TokenPerturbation",
    "EmbeddingPerturbation",
    "StructuralPerturbation",
    "SemanticPerturbation",
    "AdversarialPerturbation",
    "CompositePerturbation",
    
    # Utils - Hypervector
    "HypervectorOperations",
    "SimilarityMetrics",
    "HypervectorMemory",
    "HVEncoder",
    "HypervectorClustering",
    "HypervectorCompression",
    
    # Utils - Cryptography
    "MerkleTree",
    "CommitmentScheme",
    "HashChain",
    "PedersenCommitment",
    "SignatureScheme",
    "TimestampCommitment",
    
    # Utils - API
    "ModelAPIFactory",
    "OpenAIWrapper",
    "AnthropicWrapper",
    "HuggingFaceWrapper",
    "LocalModelWrapper",
    "BatchAPIClient",
    
    # Verification
    "FingerprintMatcher",
    "BehavioralFingerprint",
    "IncrementalFingerprint",
    "FingerprintConfig",
    "CausalGraphRecovery",
    "StructuralSimilarity",
    "HierarchicalStructure",
    "GraphConfig",
    "ZKProofSystem",
    "SchnorrProof",
    "RangeProof",
    "VectorDistanceProof",
    "MembershipProof",
    "SigmaProtocol",
    "ZKConfig",
    
    # Challenges
    "ProbeGenerator",
    "ProbeConfig",
    "ProbeDataset",
    "StandardProbeDataset",
    "AdversarialProbeDataset",
    "DomainSpecificDataset",
    "DatasetManager",
    
    # Experiments
    "ValidationExperiment",
    "BatchValidation",
    "AblationStudy",
    "PerformanceBenchmark",
]