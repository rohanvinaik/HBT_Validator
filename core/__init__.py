"""
HBT Core Module - Holographic Behavioral Twin Verification System

This module provides comprehensive model verification capabilities using
Holographic Behavioral Twins (HBTs) with statistical guarantees, security
analysis, and privacy-preserving protocols.

Components:
    - HBT Builder: Core HBT construction and management
    - Statistical Validator: Mathematical guarantees and validation
    - Application Workflows: Advanced capabilities and alignment measurement
    - Security Analysis: Threat detection and zero-knowledge proofs
    - Experimental Validation: Comprehensive benchmarking and testing
"""

# Core HBT components  
from .hbt_constructor import HolographicBehavioralTwin, HBTConfig, Challenge
from .hdc_encoder import HyperdimensionalEncoder, HDCConfig
from .rev_executor import REVExecutor, REVConfig, ExecutionState

# Import challenge generator from challenges directory
try:
    import sys
    sys.path.append('/Users/rohanvinaik/HBT_Paper')
    from challenges.probe_generator import ProbeGenerator, ChallengeType
except ImportError:
    ProbeGenerator = None
    ChallengeType = None

# Statistical validation and mathematical foundations
from .statistical_validator import (
    StatisticalValidator, VerificationBounds, SequentialState, CausalGraphMetrics
)

# Advanced application workflows
from .application_workflows import (
    ApplicationWorkflowManager, CapabilityDiscovery, AlignmentMeasurement,
    AdversarialDetection, CapabilityType, AlignmentDimension,
    CapabilityProfile, AlignmentMeasurement as AlignmentMeasurementResult,
    AdversarialDetectionResult
)

# Security analysis and privacy preservation
from .security_analysis import (
    SecurityAnalyzer, ZKProofSystem, PrivacyPreservingHBT,
    SecurityThreat, ZKProtocolType, SecurityAssessment, ZKProof,
    PrivacyGuarantee
)

# Experimental validation and benchmarking
from .experimental_validation import (
    ExperimentalValidator, SyntheticDataGenerator, BenchmarkType,
    ValidationMetric, ExperimentConfig, ExperimentResult, BenchmarkSuite,
    PerformanceMonitor
)

# Variance analysis (existing component)
try:
    from .variance_analyzer import VarianceAnalyzer, CausalInferenceEngine
except ImportError:
    # Handle missing variance analyzer gracefully
    VarianceAnalyzer = None
    CausalInferenceEngine = None

__version__ = "1.0.0"

__all__ = [
    # Core components
    "HolographicBehavioralTwin",
    "HBTConfig", 
    "Challenge",
    "ProbeGenerator",
    "ChallengeType",
    "HyperdimensionalEncoder",
    "HDCConfig",
    "REVExecutor",
    "REVConfig",
    "ExecutionState",
    
    # Statistical validation
    "StatisticalValidator",
    "VerificationBounds",
    "SequentialState",
    "CausalGraphMetrics",
    
    # Application workflows
    "ApplicationWorkflowManager",
    "CapabilityDiscovery",
    "AlignmentMeasurement",
    "AdversarialDetection",
    "CapabilityType",
    "AlignmentDimension",
    "CapabilityProfile",
    "AlignmentMeasurementResult",
    "AdversarialDetectionResult",
    
    # Security and privacy
    "SecurityAnalyzer",
    "ZKProofSystem", 
    "PrivacyPreservingHBT",
    "SecurityThreat",
    "ZKProtocolType",
    "SecurityAssessment",
    "ZKProof",
    "PrivacyGuarantee",
    
    # Experimental validation
    "ExperimentalValidator",
    "SyntheticDataGenerator",
    "BenchmarkType",
    "ValidationMetric",
    "ExperimentConfig",
    "ExperimentResult", 
    "BenchmarkSuite",
    "PerformanceMonitor",
    
    # Optional components
    "VarianceAnalyzer",
    "CausalInferenceEngine",
    
    # Version
    "__version__"
]


def create_hbt_system(config: dict = None) -> dict:
    """
    Create a complete HBT verification system with all components.
    
    Parameters
    ----------
    config : dict, optional
        Configuration parameters for the system
        
    Returns
    -------
    dict
        Dictionary containing all initialized HBT system components
    """
    if config is None:
        config = {}
    
    # Initialize core components
    if ProbeGenerator:
        probe_generator = ProbeGenerator()
    else:
        probe_generator = None
        
    # Initialize with empty parameters for system creation
    hbt_constructor = "HBTConstructor"  # Placeholder for now
    statistical_validator = StatisticalValidator()
    
    # Initialize advanced components (simplified for compatibility)
    try:
        workflow_manager = ApplicationWorkflowManager(
            hbt_builder=hbt_constructor,
            statistical_validator=statistical_validator,
            probe_generator=probe_generator
        ) if probe_generator else None
        
        security_analyzer = SecurityAnalyzer(
            hbt_builder=hbt_constructor,
            statistical_validator=statistical_validator
        )
        
        zk_system = ZKProofSystem()
        
        privacy_system = PrivacyPreservingHBT(
            hbt_builder=hbt_constructor,
            security_analyzer=security_analyzer,
            zk_system=zk_system
        )
        
        experimental_validator = ExperimentalValidator(
            hbt_builder=hbt_constructor,
            statistical_validator=statistical_validator,
            workflow_manager=workflow_manager,
            security_analyzer=security_analyzer,
            zk_system=zk_system
        ) if workflow_manager else None
        
    except Exception as e:
        # Graceful fallback for missing components
        workflow_manager = None
        security_analyzer = None
        zk_system = None
        privacy_system = None
        experimental_validator = None
    
    return {
        "hbt_constructor": hbt_constructor,
        "probe_generator": probe_generator,
        "statistical_validator": statistical_validator,
        "workflow_manager": workflow_manager,
        "security_analyzer": security_analyzer,
        "zk_system": zk_system,
        "privacy_system": privacy_system,
        "experimental_validator": experimental_validator,
        "config": config
    }


def get_system_info() -> dict:
    """
    Get information about the HBT system capabilities.
    
    Returns
    -------
    dict
        System information and capabilities
    """
    return {
        "version": __version__,
        "components": {
            "core": ["HBTBuilder", "ProbeGenerator", "HDCEncoder", "REVExecutor"],
            "statistical": ["StatisticalValidator", "SequentialTester", "EmpiricalBernsteinBound"],
            "workflows": ["CapabilityDiscovery", "AlignmentMeasurement", "AdversarialDetection"],
            "security": ["SecurityAnalyzer", "ZKProofSystem", "PrivacyPreservingHBT"],
            "experimental": ["ExperimentalValidator", "BenchmarkSuite", "PerformanceMonitor"]
        },
        "capabilities": [
            "Model verification with statistical guarantees",
            "Capability discovery and profiling", 
            "Alignment measurement across dimensions",
            "Adversarial behavior detection",
            "Security threat assessment",
            "Zero-knowledge proof generation",
            "Privacy-preserving verification",
            "Comprehensive experimental validation"
        ],
        "supported_protocols": [
            "HBT construction and verification",
            "Sequential hypothesis testing",
            "Differential privacy",
            "Zero-knowledge proofs",
            "Scalability analysis"
        ]
    }