"""
Base classes for election data parsers.
All parsers should inherit from BaseParser and return ElectionData objects.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod


@dataclass
class ElectionData:
    """
    Standard format for election data to feed into compare_ks.py
    """
    name: str                          # Election name (for output directory)
    candidates: List[str]              # List of candidate names
    bound_ballots: List[List[str]]     # Observed/bound ballots (partial rankings)
    n_bound: int                       # Number of bound ballots
    U: int                             # Number of real unbound ballots (content treated as unknown)
    P: Dict[str, float]                # Prior probabilities for each candidate
    metadata: Dict                     # Additional election info (year, location, etc.)
    unbound_ballots: Optional[List[List[str]]] = None  # Actual unbound ballots (if available)
    
    def __post_init__(self):
        """Validate the data"""
        assert len(self.candidates) >= 2, "Need at least 2 candidates"
        assert self.n_bound == len(self.bound_ballots), "n_bound must match ballot count"
        assert abs(sum(self.P.values()) - 1.0) < 1e-6, f"P must sum to 1, got {sum(self.P.values())}"
        assert set(self.P.keys()) == set(self.candidates), "P must have all candidates"
        if self.unbound_ballots is not None:
            assert len(self.unbound_ballots) == self.U, "unbound_ballots length must match U"
    
    def summary(self) -> str:
        """Human-readable summary"""
        return (
            f"Election: {self.name}\n"
            f"Candidates: {', '.join(self.candidates)}\n"
            f"Bound ballots: {self.n_bound}\n"
            f"Unbound to simulate: {self.U}\n"
            f"Prior P: {self.P}\n"
            f"Metadata: {self.metadata}"
        )


class BaseParser(ABC):
    """
    Abstract base class for election data parsers.
    """
    
    @abstractmethod
    def can_parse(self, filepath: str) -> bool:
        """
        Check if this parser can handle the given file.
        Returns True if it can parse, False otherwise.
        """
        pass
    
    @abstractmethod
    def parse(self, filepath: str, **kwargs) -> ElectionData:
        """
        Parse the election file and return ElectionData.

        Real-data contract: bound/unbound classification comes exclusively
        from explicit markings in the file (ballot-type / counting-group
        columns); U is the count of real unbound ballots and P is derived
        from bound first choices. No synthetic splitting or generation.
        (**kwargs is reserved for parser-specific options; the legacy
        U_method/U_value/P_method knobs of the removed synthetic-split
        parsers are gone.)
        """
        pass
    
    @staticmethod
    def compute_first_choice_probs(ballots: List[List[str]],
                                   candidates: List[str]) -> Dict[str, float]:
        counts = {c: 0 for c in candidates}
        total = 0
        for ballot in ballots:
            if ballot:
                counts[ballot[0]] += 1
                total += 1
        
        if total == 0:
            return {c: 1.0/len(candidates) for c in candidates}
        
        return {c: counts[c]/total for c in candidates}
