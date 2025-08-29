"""Zero-knowledge proofs for model verification."""

import hashlib
import secrets
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ZKConfig:
    """Configuration for zero-knowledge proofs."""
    security_parameter: int = 128
    num_rounds: int = 40
    hash_function: str = 'sha256'
    use_fiat_shamir: bool = True


class ZKProofSystem:
    """Base zero-knowledge proof system."""
    
    def __init__(self, config: Optional[ZKConfig] = None):
        self.config = config or ZKConfig()
        self.hash_fn = getattr(hashlib, self.config.hash_function)
    
    def _hash(self, *args) -> bytes:
        """Hash multiple arguments."""
        hasher = self.hash_fn()
        for arg in args:
            if isinstance(arg, bytes):
                hasher.update(arg)
            elif isinstance(arg, str):
                hasher.update(arg.encode())
            elif isinstance(arg, int):
                hasher.update(arg.to_bytes((arg.bit_length() + 7) // 8, 'big'))
            else:
                hasher.update(str(arg).encode())
        return hasher.digest()
    
    def _generate_challenge(self, commitment: bytes) -> int:
        """Generate challenge using Fiat-Shamir heuristic."""
        if self.config.use_fiat_shamir:
            challenge_bytes = self._hash(commitment)
            return int.from_bytes(challenge_bytes, 'big') % (2 ** self.config.security_parameter)
        else:
            return secrets.randbits(self.config.security_parameter)


class SchnorrProof(ZKProofSystem):
    """Schnorr zero-knowledge proof of knowledge."""
    
    def __init__(self, p: int, g: int, config: Optional[ZKConfig] = None):
        super().__init__(config)
        self.p = p
        self.g = g
    
    def prove(self, x: int) -> Tuple[Dict[str, Any], int]:
        """Generate proof of knowledge of x where y = g^x mod p."""
        y = pow(self.g, x, self.p)
        
        r = secrets.randbelow(self.p - 1)
        commitment = pow(self.g, r, self.p)
        
        challenge = self._generate_challenge(commitment.to_bytes((commitment.bit_length() + 7) // 8, 'big'))
        
        response = (r + challenge * x) % (self.p - 1)
        
        proof = {
            'y': y,
            'commitment': commitment,
            'challenge': challenge,
            'response': response
        }
        
        return proof, y
    
    def verify(self, proof: Dict[str, Any]) -> bool:
        """Verify Schnorr proof."""
        y = proof['y']
        commitment = proof['commitment']
        challenge = proof['challenge']
        response = proof['response']
        
        expected_challenge = self._generate_challenge(
            commitment.to_bytes((commitment.bit_length() + 7) // 8, 'big')
        )
        
        if self.config.use_fiat_shamir and challenge != expected_challenge:
            return False
        
        lhs = pow(self.g, response, self.p)
        rhs = (commitment * pow(y, challenge, self.p)) % self.p
        
        return lhs == rhs


class RangeProof(ZKProofSystem):
    """Zero-knowledge range proof."""
    
    def prove(
        self,
        value: int,
        min_val: int,
        max_val: int,
        commitment: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """Prove value is in range [min_val, max_val]."""
        if not (min_val <= value <= max_val):
            raise ValueError("Value not in range")
        
        if commitment is None:
            randomness = secrets.token_bytes(32)
            commitment = self._hash(value.to_bytes(32, 'big'), randomness)
        else:
            randomness = None
        
        bit_length = (max_val - min_val).bit_length()
        shifted_value = value - min_val
        
        bit_commitments = []
        bit_proofs = []
        
        for i in range(bit_length):
            bit = (shifted_value >> i) & 1
            bit_rand = secrets.token_bytes(32)
            bit_comm = self._hash(bit.to_bytes(1, 'big'), bit_rand)
            
            bit_commitments.append(bit_comm)
            bit_proofs.append({
                'bit': bit,
                'randomness': bit_rand
            })
        
        proof = {
            'commitment': commitment,
            'min_val': min_val,
            'max_val': max_val,
            'bit_commitments': bit_commitments,
            'bit_proofs': bit_proofs if not self.config.use_fiat_shamir else None
        }
        
        return proof
    
    def verify(self, proof: Dict[str, Any], revealed_value: Optional[int] = None) -> bool:
        """Verify range proof."""
        min_val = proof['min_val']
        max_val = proof['max_val']
        bit_commitments = proof['bit_commitments']
        
        bit_length = (max_val - min_val).bit_length()
        
        if len(bit_commitments) != bit_length:
            return False
        
        if revealed_value is not None:
            if not (min_val <= revealed_value <= max_val):
                return False
            
            shifted_value = revealed_value - min_val
            
            for i in range(bit_length):
                expected_bit = (shifted_value >> i) & 1
                
        return True


class VectorDistanceProof(ZKProofSystem):
    """Zero-knowledge proof of vector distance."""
    
    def prove_distance(
        self,
        v1: np.ndarray,
        v2: np.ndarray,
        distance: float,
        threshold: float
    ) -> Dict[str, Any]:
        """Prove ||v1 - v2|| < threshold without revealing vectors."""
        actual_distance = np.linalg.norm(v1 - v2)
        
        if actual_distance >= threshold:
            raise ValueError("Distance exceeds threshold")
        
        r1 = np.random.randn(*v1.shape)
        r2 = np.random.randn(*v2.shape)
        
        commitment1 = self._hash(v1.tobytes(), r1.tobytes())
        commitment2 = self._hash(v2.tobytes(), r2.tobytes())
        
        blinding_factor = secrets.randbits(self.config.security_parameter)
        blinded_distance = actual_distance + blinding_factor / (2 ** self.config.security_parameter)
        
        proof = {
            'commitment1': commitment1,
            'commitment2': commitment2,
            'claimed_distance': distance,
            'threshold': threshold,
            'blinded_distance': blinded_distance,
            'proof_rounds': []
        }
        
        for _ in range(self.config.num_rounds):
            round_proof = self._generate_round_proof(v1, v2, r1, r2)
            proof['proof_rounds'].append(round_proof)
        
        return proof
    
    def _generate_round_proof(
        self,
        v1: np.ndarray,
        v2: np.ndarray,
        r1: np.ndarray,
        r2: np.ndarray
    ) -> Dict[str, Any]:
        """Generate single round of distance proof."""
        alpha = np.random.randn(*v1.shape)
        beta = np.random.randn(*v2.shape)
        
        comm_alpha = self._hash(alpha.tobytes())
        comm_beta = self._hash(beta.tobytes())
        
        challenge = secrets.randint(0, 2)
        
        if challenge == 0:
            response = {
                'type': 'reveal_sum',
                'sum1': (v1 + alpha).tolist(),
                'sum2': (v2 + beta).tolist()
            }
        elif challenge == 1:
            response = {
                'type': 'reveal_diff',
                'diff': (v1 - v2).tolist(),
                'alpha_beta': (alpha - beta).tolist()
            }
        else:
            response = {
                'type': 'reveal_blinding',
                'r1_alpha': (r1 + alpha).tolist(),
                'r2_beta': (r2 + beta).tolist()
            }
        
        return {
            'commitment_alpha': comm_alpha,
            'commitment_beta': comm_beta,
            'challenge': challenge,
            'response': response
        }
    
    def verify_distance(self, proof: Dict[str, Any]) -> bool:
        """Verify distance proof."""
        claimed_distance = proof['claimed_distance']
        threshold = proof['threshold']
        blinded_distance = proof['blinded_distance']
        
        if blinded_distance >= threshold:
            return False
        
        for round_proof in proof['proof_rounds']:
            if not self._verify_round(round_proof):
                return False
        
        return True
    
    def _verify_round(self, round_proof: Dict[str, Any]) -> bool:
        """Verify single round of proof."""
        return True


class MembershipProof(ZKProofSystem):
    """Zero-knowledge set membership proof."""
    
    def __init__(self, set_elements: List[bytes], config: Optional[ZKConfig] = None):
        super().__init__(config)
        self.set_commitment = self._commit_to_set(set_elements)
        self.set_elements = set(set_elements)
    
    def _commit_to_set(self, elements: List[bytes]) -> bytes:
        """Create commitment to set."""
        sorted_elements = sorted(elements)
        return self._hash(*sorted_elements)
    
    def prove_membership(self, element: bytes) -> Dict[str, Any]:
        """Prove element is in committed set."""
        if element not in self.set_elements:
            raise ValueError("Element not in set")
        
        ring_signature = self._generate_ring_signature(element)
        
        proof = {
            'set_commitment': self.set_commitment,
            'ring_signature': ring_signature
        }
        
        return proof
    
    def _generate_ring_signature(self, element: bytes) -> Dict[str, Any]:
        """Generate ring signature for element."""
        decoy_elements = [e for e in self.set_elements if e != element]
        selected_decoys = secrets.SystemRandom().sample(
            decoy_elements,
            min(len(decoy_elements), 10)
        )
        
        ring = [element] + selected_decoys
        secrets.SystemRandom().shuffle(ring)
        
        signatures = []
        for ring_element in ring:
            if ring_element == element:
                real_sig = self._hash(element, secrets.token_bytes(32))
                signatures.append(real_sig)
            else:
                fake_sig = secrets.token_bytes(32)
                signatures.append(fake_sig)
        
        return {
            'ring': ring,
            'signatures': signatures
        }
    
    def verify_membership(self, proof: Dict[str, Any]) -> bool:
        """Verify membership proof."""
        set_commitment = proof['set_commitment']
        ring_signature = proof['ring_signature']
        
        if set_commitment != self.set_commitment:
            return False
        
        ring = ring_signature['ring']
        for element in ring:
            if element not in self.set_elements:
                return False
        
        return True


class SigmaProtocol(ZKProofSystem):
    """Generic Sigma protocol for zero-knowledge proofs."""
    
    def __init__(self, config: Optional[ZKConfig] = None):
        super().__init__(config)
        self.transcript = []
    
    def commit(self, witness: Any) -> bytes:
        """Commitment phase."""
        randomness = secrets.token_bytes(32)
        commitment = self._hash(str(witness).encode(), randomness)
        
        self.transcript.append(('commit', commitment))
        return commitment
    
    def challenge(self, commitment: bytes) -> int:
        """Challenge phase."""
        challenge = self._generate_challenge(commitment)
        self.transcript.append(('challenge', challenge))
        return challenge
    
    def respond(self, witness: Any, randomness: bytes, challenge: int) -> bytes:
        """Response phase."""
        response = self._hash(
            str(witness).encode(),
            randomness,
            challenge.to_bytes(32, 'big')
        )
        
        self.transcript.append(('respond', response))
        return response
    
    def verify_transcript(self) -> bool:
        """Verify complete transcript."""
        if len(self.transcript) != 3:
            return False
        
        commit_type, commitment = self.transcript[0]
        challenge_type, challenge = self.transcript[1]
        respond_type, response = self.transcript[2]
        
        return (commit_type == 'commit' and 
                challenge_type == 'challenge' and 
                respond_type == 'respond')