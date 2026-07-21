"""
Main parser coordinator that detects formats and delegates to appropriate parser.
"""

import os
from .formats.base_parser import ElectionData
from .formats.strict_parser import StrictRealBallotParser


class ElectionDataParser:
    """
    Main parser that auto-detects format and delegates to appropriate parser.

    CSV inputs go through StrictRealBallotParser, which hard-fails on any
    unrecognized header layout or any unrecognized ballot-type value rather
    than silently classifying or generating synthetic data.
    """

    def __init__(self):
        self.parsers = [
            StrictRealBallotParser(),
        ]
    
    def parse(self, filepath: str, **kwargs) -> ElectionData:
        """
        Parse election data file, auto-detecting format.
        
        Args:
            filepath: Path to the election data file
            **kwargs: Parser-specific options (see individual parsers)
        
        Returns:
            ElectionData object ready for compare_ks.py
        
        Raises:
            ValueError: If no parser can handle the file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Try each parser
        for parser in self.parsers:
            if parser.can_parse(filepath):
                print(f"Using {parser.__class__.__name__} for {filepath}")
                return parser.parse(filepath, **kwargs)
        
        # No parser found
        raise ValueError(
            f"No parser found for file: {filepath}\n"
            f"Supported formats: .csv (strict real-ballot layout)\n"
            f"Please implement a custom parser or convert your data."
        )
    

# Convenience function
def parse_election(filepath: str, **kwargs) -> ElectionData:
    """
    Parse a single election file.

    Example:
        election = parse_election('data/dataverse_files/Alaska_20221108_HouseDistrict18.csv')
    """
    parser = ElectionDataParser()
    return parser.parse(filepath, **kwargs)
