"""HBT Validator - Hypervector Behavioral Tree Validator for LLM Verification."""

__version__ = "0.1.0"

# Core components
from core.hbt_constructor import HolographicBehavioralTwin, HBTConfig
from core.hdc_encoder import HyperdimensionalEncoder, HDCConfig
from core.variance_analyzer import VarianceAnalyzer, VarianceConfig
from core.rev_executor import REVExecutor
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
    OpenAIAPI,
    AnthropicAPI,
    LocalModelAPI,
    BatchedAPIClient,
    BaseModelAPI,
    ModelResponse
)

# Verification
from verification.fingerprint_matcher import (
    FingerprintMatcher,
    BehavioralFingerprint,
    FingerprintConfig,
    VerificationResult,
    ZKProof,
    LineageTracker
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
from challenges.probe_generator import ProbeGenerator, Challenge
# from challenges.datasets import (
#     ProbeDataset,
#     StandardProbeDataset,
#     AdversarialProbeDataset,
#     DomainSpecificDataset,
#     DatasetManager
# )

# Experiments
# from experiments.validation import ValidationExperiment, BatchValidation
# from experiments.ablations import AblationStudy
# from experiments.benchmarks import PerformanceBenchmark

__all__ = [
    # Version
    "__version__",
    
    # Core
    "HolographicBehavioralTwin",
    "HBTConfig",
    "HyperdimensionalEncoder",
    "HDCConfig",
    "VarianceAnalyzer",
    "VarianceConfig",
    "REVExecutor",
    "REVExecutorEnhanced", 
    "REVConfig",
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
    "OpenAIAPI", 
    "AnthropicAPI",
    "LocalModelAPI",
    "BatchedAPIClient",
    "BaseModelAPI",
    "ModelResponse",
    
    # Verification
    "FingerprintMatcher",
    "BehavioralFingerprint", 
    "FingerprintConfig",
    "VerificationResult",
    "ZKProof",
    "LineageTracker",
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
    "Challenge",
    # "ProbeDataset",
    # "StandardProbeDataset",
    # "AdversarialProbeDataset",
    # "DomainSpecificDataset",
    # "DatasetManager",
    
    # Experiments
    # "ValidationExperiment",
    # "BatchValidation",
    # "AblationStudy",
    # "PerformanceBenchmark",
]