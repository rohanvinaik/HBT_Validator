"""HBT Validator - Hypervector Behavioral Tree Validator for LLM Verification."""

__version__ = "0.1.0"

# Core components
from core.hbt_constructor import HBTConstructor, HBTConfig
from core.hdc_encoder import HyperdimensionalEncoder, HDCConfig
from core.variance_analyzer import VarianceAnalyzer, VarianceConfig
from core.rev_executor import REVExecutor, SegmentConfig
from core.rev_executor_enhanced import (
    REVExecutor as REVExecutorEnhanced,
    REVConfig,
    SegmentSignature,
    MerkleNode
)

# Utils
from utils.perturbations import (
    PerturbationOperator,
    TokenPerturbation,
    EmbeddingPerturbation,
    StructuralPerturbation,
    SemanticPerturbation,
    AdversarialPerturbation,
    CompositePerturbation
)
from utils.hypervector_ops import (
    HypervectorOperations,
    SimilarityMetrics,
    HypervectorMemory,
    HypervectorEncoder as HVEncoder,
    HypervectorClustering,
    HypervectorCompression
)
from utils.cryptography import (
    MerkleTree,
    CommitmentScheme,
    HashChain,
    PedersenCommitment,
    SignatureScheme,
    TimestampCommitment
)
from utils.api_wrappers import (
    ModelAPIFactory,
    OpenAIWrapper,
    AnthropicWrapper,
    HuggingFaceWrapper,
    LocalModelWrapper,
    BatchAPIClient
)

# Verification
from verification.fingerprint_matcher import (
    FingerprintMatcher,
    BehavioralFingerprint,
    IncrementalFingerprint,
    FingerprintConfig
)
from verification.structural_inference import (
    CausalGraphRecovery,
    StructuralSimilarity,
    HierarchicalStructure,
    GraphConfig
)
from verification.zk_proofs import (
    ZKProofSystem,
    SchnorrProof,
    RangeProof,
    VectorDistanceProof,
    MembershipProof,
    SigmaProtocol,
    ZKConfig
)

# Challenges
from challenges.probe_generator import ProbeGenerator, ProbeConfig
from challenges.datasets import (
    ProbeDataset,
    StandardProbeDataset,
    AdversarialProbeDataset,
    DomainSpecificDataset,
    DatasetManager
)

# Experiments
from experiments.validation import ValidationExperiment, BatchValidation
from experiments.ablations import AblationStudy
from experiments.benchmarks import PerformanceBenchmark

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