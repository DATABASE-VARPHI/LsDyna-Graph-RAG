#!/usr/bin/env python3
"""
LS-DYNA Graph RAG - Parser Module
--------------------------------
This module uses pydyna to parse LS-DYNA input files and generates LISP expressions
for ontology graph generation.
"""

import os
import sys
import argparse
from typing import Dict, List, Any, Tuple, Optional
import json

try:
    import pydyna
except ImportError:
    print("Error: pydyna package not found. Please install it using:")
    print("pip install pydyna")
    sys.exit(1)

class LSDynaParser:
    """Parser for LS-DYNA input files using pydyna, with LISP expression output."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the parser.
        
        Args:
            verbose: Flag to enable verbose output
        """
        self.verbose = verbose
        self.keywords = {}
        self.entities = {}
        self.relationships = []
    
    def parse_file(self, input_file: str) -> Dict[str, Any]:
        """Parse an LS-DYNA input file using pydyna.
        
        Args:
            input_file: Path to the LS-DYNA input file
            
        Returns:
            Dict containing the parsed data structure
        """
        if self.verbose:
            print(f"Parsing {input_file}...")
        
        # Use pydyna to parse the input file
        try:
            model = pydyna.read_input_file(input_file)
            
            # Extract keywords from the model
            for keyword in model.get_keywords():
                keyword_type = keyword.get_type()
                self.keywords[keyword_type] = []
                
                # Extract cards for each keyword
                for card in keyword.get_cards():
                    card_data = {
                        "id": card.get_id() if hasattr(card, "get_id") else None,
                        "parameters": card.get_parameters(),
                        "fields": [field for field in card.get_fields()]
                    }
                    self.keywords[keyword_type].append(card_data)
                    
                    # Create entity from card
                    entity_id = card_data["id"] if card_data["id"] else f"{keyword_type}_{len(self.entities)}"
                    self.entities[entity_id] = {
                        "type": keyword_type,
                        "properties": card_data
                    }
            
            # Process relationships between entities
            self._process_relationships()
            
            return {
                "keywords": self.keywords,
                "entities": self.entities,
                "relationships": self.relationships
            }
            
        except Exception as e:
            print(f"Error parsing file {input_file}: {str(e)}")
            return {}
    
    def _process_relationships(self):
        """Process relationships between entities based on keyword references."""
        
        # Map of keyword types that typically reference other entities
        reference_mappings = {
            "*PART": ["*SECTION", "*MAT"],
            "*ELEMENT_SHELL": ["*PART"],
            "*ELEMENT_SOLID": ["*PART"],
            "*ELEMENT_BEAM": ["*PART"],
            "*CONTACT": ["*PART"],
            "*BOUNDARY": ["*NODE"],
            # Add more mappings as needed
        }
        
        # Extract relationships based on reference mappings
        for entity_id, entity in self.entities.items():
            entity_type = entity["type"]
            
            if entity_type in reference_mappings:
                for ref_type in reference_mappings[entity_type]:
                    # Look for references to other entities in properties
                    if "properties" in entity and "parameters" in entity["properties"]:
                        for param_name, param_value in entity["properties"]["parameters"].items():
                            if isinstance(param_value, str) and param_value in self.entities:
                                if self.entities[param_value]["type"] == ref_type:
                                    self.relationships.append({
                                        "source": entity_id,
                                        "target": param_value,
                                        "type": f"REFERENCES_{ref_type.replace('*', '')}"
                                    })
    
    def generate_lisp(self) -> str:
        """Generate LISP expressions from the parsed data.
        
        Returns:
            String containing LISP expressions representing the model
        """
        lisp_expressions = []
        
        # Generate entity expressions
        for entity_id, entity in self.entities.items():
            entity_type = entity["type"].replace("*", "")
            props = []
            
            if "properties" in entity:
                # Add core properties
                if "parameters" in entity["properties"]:
                    for key, value in entity["properties"]["parameters"].items():
                        props.append(f":{key} {self._lisp_value(value)}")
                
                # Add fields if available
                if "fields" in entity["properties"]:
                    for i, field in enumerate(entity["properties"]["fields"]):
                        if field is not None:
                            props.append(f":field{i} {self._lisp_value(field)}")
            
            # Create entity expression
            entity_expr = f"(def-entity {entity_id} {entity_type} {' '.join(props)})"
            lisp_expressions.append(entity_expr)
        
        # Generate relationship expressions
        for rel in self.relationships:
            rel_expr = f"(def-relation {rel['source']} {rel['type']} {rel['target']})"
            lisp_expressions.append(rel_expr)
        
        return "\n".join(lisp_expressions)
    
    def _lisp_value(self, value: Any) -> str:
        """Convert a Python value to a LISP-compatible string representation.
        
        Args:
            value: The value to convert
            
        Returns:
            LISP string representation of the value
        """
        if value is None:
            return "nil"
        elif isinstance(value, bool):
            return "t" if value else "nil"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            # Escape quotes and special characters in strings
            escaped = value.replace('"', '\\"')
            return f'"{escaped}"'
        elif isinstance(value, list):
            items = " ".join(self._lisp_value(item) for item in value)
            return f"({items})"
        elif isinstance(value, dict):
            items = " ".join([f":{k} {self._lisp_value(v)}" for k, v in value.items()])
            return f"({items})"
        else:
            return str(value)
    
    def save_lisp(self, output_file: str):
        """Save LISP expressions to a file.
        
        Args:
            output_file: Path to the output file
        """
        lisp_content = self.generate_lisp()
        
        try:
            with open(output_file, 'w') as f:
                f.write(lisp_content)
            
            if self.verbose:
                print(f"LISP expressions saved to {output_file}")
        
        except Exception as e:
            print(f"Error saving LISP to {output_file}: {str(e)}")


def main():
    """Main function to run the parser from command line."""
    parser = argparse.ArgumentParser(description="Parse LS-DYNA input files and generate LISP expressions")
    parser.add_argument("input_file", help="Path to the LS-DYNA input file")
    parser.add_argument("-o", "--output", help="Output file for LISP expressions", default="output.lisp")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        sys.exit(1)
    
    # Parse the file and generate LISP
    lsdyna_parser = LSDynaParser(verbose=args.verbose)
    parsed_data = lsdyna_parser.parse_file(args.input_file)
    
    if parsed_data:
        # Save LISP expressions
        lsdyna_parser.save_lisp(args.output)
        
        print(f"Parsing completed successfully!")
        print(f"Found {len(parsed_data['entities'])} entities and {len(parsed_data['relationships'])} relationships.")
        print(f"LISP expressions saved to {args.output}")


if __name__ == "__main__":
    main()
