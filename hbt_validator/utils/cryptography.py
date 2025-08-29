"""Cryptographic utilities for commitments and proofs."""

import hashlib
import hmac
import secrets
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MerkleTree:
    """Merkle tree implementation for commitment schemes."""
    
    def __init__(self, hash_fn: str = 'sha256'):
        self.hash_fn = getattr(hashlib, hash_fn)
        self.tree = []
        self.leaves = []
        self.root = None
    
    def build(self, data_blocks: List[bytes]) -> bytes:
        """Build Merkle tree from data blocks."""
        if not data_blocks:
            return b''
        
        self.leaves = [self._hash(block) for block in data_blocks]
        self.tree = [self.leaves]
        
        current_level = self.leaves
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    combined = current_level[i] + current_level[i + 1]
                else:
                    combined = current_level[i] + current_level[i]
                next_level.append(self._hash(combined))
            
            self.tree.append(next_level)
            current_level = next_level
        
        self.root = current_level[0] if current_level else b''
        return self.root
    
    def get_proof(self, index: int) -> List[Tuple[bytes, str]]:
        """Get Merkle proof for a leaf at given index."""
        if index >= len(self.leaves):
            raise IndexError(f"Index {index} out of range")
        
        proof = []
        for level in self.tree[:-1]:
            if index % 2 == 0:
                if index + 1 < len(level):
                    proof.append((level[index + 1], 'right'))
            else:
                proof.append((level[index - 1], 'left'))
            
            index //= 2
        
        return proof
    
    def verify_proof(
        self,
        leaf_data: bytes,
        proof: List[Tuple[bytes, str]],
        root: bytes
    ) -> bool:
        """Verify Merkle proof."""
        current = self._hash(leaf_data)
        
        for sibling, direction in proof:
            if direction == 'left':
                combined = sibling + current
            else:
                combined = current + sibling
            current = self._hash(combined)
        
        return current == root
    
    def _hash(self, data: bytes) -> bytes:
        """Hash data using configured hash function."""
        return self.hash_fn(data).digest()


class CommitmentScheme:
    """Cryptographic commitment schemes."""
    
    def __init__(self, security_bits: int = 256):
        self.security_bits = security_bits
    
    def commit(self, data: bytes) -> Tuple[bytes, bytes]:
        """Create commitment with randomness."""
        randomness = secrets.token_bytes(self.security_bits // 8)
        
        commitment = hashlib.sha3_256(data + randomness).digest()
        
        return commitment, randomness
    
    def verify(
        self,
        commitment: bytes,
        data: bytes,
        randomness: bytes
    ) -> bool:
        """Verify commitment."""
        expected = hashlib.sha3_256(data + randomness).digest()
        return hmac.compare_digest(commitment, expected)
    
    def vector_commit(
        self,
        vectors: List[np.ndarray]
    ) -> Tuple[bytes, List[bytes]]:
        """Commit to multiple vectors."""
        commitments = []
        randomness_list = []
        
        for vector in vectors:
            vector_bytes = vector.tobytes()
            comm, rand = self.commit(vector_bytes)
            commitments.append(comm)
            randomness_list.append(rand)
        
        tree = MerkleTree()
        root = tree.build(commitments)
        
        return root, randomness_list


class HashChain:
    """Hash chain for sequential commitments."""
    
    def __init__(self, seed: bytes, length: int):
        self.seed = seed
        self.length = length
        self.chain = self._generate_chain()
    
    def _generate_chain(self) -> List[bytes]:
        """Generate hash chain."""
        chain = []
        current = self.seed
        
        for _ in range(self.length):
            current = hashlib.sha256(current).digest()
            chain.append(current)
        
        return chain
    
    def get_value(self, index: int) -> bytes:
        """Get value at index in chain."""
        if 0 <= index < self.length:
            return self.chain[index]
        raise IndexError(f"Index {index} out of range")
    
    def verify_link(self, value: bytes, next_value: bytes) -> bool:
        """Verify link in hash chain."""
        expected = hashlib.sha256(value).digest()
        return hmac.compare_digest(expected, next_value)


class PedersenCommitment:
    """Pedersen commitment scheme for vectors."""
    
    def __init__(self, modulus: int = None):
        self.modulus = modulus or self._generate_safe_prime()
        self.g = self._find_generator()
        self.h = self._find_generator()
    
    def _generate_safe_prime(self, bits: int = 256) -> int:
        """Generate safe prime for cryptographic use."""
        from sympy import nextprime, isprime
        
        while True:
            q = nextprime(secrets.randbits(bits - 1))
            p = 2 * q + 1
            if isprime(p):
                return p
    
    def _find_generator(self) -> int:
        """Find generator for multiplicative group."""
        while True:
            g = secrets.randbelow(self.modulus - 2) + 2
            if pow(g, 2, self.modulus) != 1 and \
               pow(g, (self.modulus - 1) // 2, self.modulus) != 1:
                return g
    
    def commit(self, value: int, randomness: Optional[int] = None) -> Tuple[int, int]:
        """Create Pedersen commitment."""
        if randomness is None:
            randomness = secrets.randbelow(self.modulus)
        
        commitment = (pow(self.g, value, self.modulus) * 
                     pow(self.h, randomness, self.modulus)) % self.modulus
        
        return commitment, randomness
    
    def verify(
        self,
        commitment: int,
        value: int,
        randomness: int
    ) -> bool:
        """Verify Pedersen commitment."""
        expected = (pow(self.g, value, self.modulus) * 
                   pow(self.h, randomness, self.modulus)) % self.modulus
        
        return commitment == expected
    
    def add_commitments(
        self,
        comm1: int,
        comm2: int
    ) -> int:
        """Add two commitments homomorphically."""
        return (comm1 * comm2) % self.modulus


class SignatureScheme:
    """Digital signature utilities."""
    
    @staticmethod
    def sign_data(data: bytes, private_key: bytes) -> bytes:
        """Sign data with private key."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa, padding
        from cryptography.hazmat.primitives import serialization
        
        try:
            key = serialization.load_pem_private_key(private_key, password=None)
            signature = key.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return signature
        except Exception as e:
            logger.error(f"Signing failed: {e}")
            return b''
    
    @staticmethod
    def verify_signature(
        data: bytes,
        signature: bytes,
        public_key: bytes
    ) -> bool:
        """Verify signature with public key."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa, padding
        from cryptography.hazmat.primitives import serialization
        
        try:
            key = serialization.load_pem_public_key(public_key)
            key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False


class TimestampCommitment:
    """Time-locked commitments."""
    
    def __init__(self, difficulty: int = 20):
        self.difficulty = difficulty
    
    def create_puzzle(self, data: bytes) -> Dict[str, Any]:
        """Create time-lock puzzle."""
        nonce = 0
        target = 2 ** (256 - self.difficulty)
        
        while True:
            candidate = hashlib.sha256(data + nonce.to_bytes(8, 'big')).digest()
            if int.from_bytes(candidate, 'big') < target:
                return {
                    'data': data,
                    'nonce': nonce,
                    'hash': candidate,
                    'difficulty': self.difficulty
                }
            nonce += 1
    
    def verify_puzzle(self, puzzle: Dict[str, Any]) -> bool:
        """Verify time-lock puzzle solution."""
        target = 2 ** (256 - puzzle['difficulty'])
        
        candidate = hashlib.sha256(
            puzzle['data'] + puzzle['nonce'].to_bytes(8, 'big')
        ).digest()
        
        return (int.from_bytes(candidate, 'big') < target and 
                candidate == puzzle['hash'])