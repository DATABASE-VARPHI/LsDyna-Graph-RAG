#!/usr/bin/env python3
"""
LS-DYNA Graph RAG - Graph Builder Module
---------------------------------------
This module takes LISP expressions generated from LS-DYNA files and builds a knowledge graph.
"""

import os
import sys
import argparse
import re
from typing import Dict, List, Any, Tuple, Set
import networkx as nx
import matplotlib.pyplot as plt
import json

class LispParser:
    """Parser for LISP expressions generated from LS-DYNA files."""
    
    def __init__(self):
        """Initialize the LISP parser."""
        self.entities = {}
        self.relations = []
        
    def parse_file(self, lisp_file: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Parse a file containing LISP expressions.
        
        Args:
            lisp_file: Path to the file containing LISP expressions
            
        Returns:
            Tuple containing entities and relations
        """
        try:
            with open(lisp_file, 'r') as f:
                content = f.read()
                
            # Parse entities and relations
            self._parse_entities(content)
            self._parse_relations(content)
            
            return self.entities, self.relations
            
        except Exception as e:
            print(f"Error parsing LISP file {lisp_file}: {str(e)}")
            return {}, []
    
    def _parse_entities(self, content: str):
        """Parse entity definitions from LISP content.
        
        Args:
            content: String containing LISP expressions
        """
        # Match def-entity expressions
        entity_pattern = r'\(def-entity\s+([^\s]+)\s+([^\s]+)(.*?)\)'
        entity_matches = re.finditer(entity_pattern, content, re.DOTALL)
        
        for match in entity_matches:
            entity_id = match.group(1).strip()
            entity_type = match.group(2).strip()
            properties_str = match.group(3).strip()
            
            # Parse properties
            properties = {}
            prop_pattern = r':([^\s]+)\s+([^\s:]+|".*?")'
            prop_matches = re.finditer(prop_pattern, properties_str)
            
            for prop_match in prop_matches:
                prop_name = prop_match.group(1).strip()
                prop_value = prop_match.group(2).strip()
                
                # Remove quotes from string values
                if prop_value.startswith('"') and prop_value.endswith('"'):
                    prop_value = prop_value[1:-1].replace('\\"', '"')
                
                # Convert numeric values
                if re.match(r'^-?\d+$', prop_value):
                    prop_value = int(prop_value)
                elif re.match(r'^-?\d+\.\d*$', prop_value):
                    prop_value = float(prop_value)
                
                properties[prop_name] = prop_value
            
            self.entities[entity_id] = {
                'type': entity_type,
                'properties': properties
            }
    
    def _parse_relations(self, content: str):
        """Parse relation definitions from LISP content.
        
        Args:
            content: String containing LISP expressions
        """
        # Match def-relation expressions
        relation_pattern = r'\(def-relation\s+([^\s]+)\s+([^\s]+)\s+([^\s\)]+)\)'
        relation_matches = re.finditer(relation_pattern, content)
        
        for match in relation_matches:
            source = match.group(1).strip()
            rel_type = match.group(2).strip()
            target = match.group(3).strip()
            
            self.relations.append({
                'source': source,
                'type': rel_type,
                'target': target
            })


class GraphBuilder:
    """Builds a knowledge graph from parsed LS-DYNA data."""
    
    def __init__(self):
        """Initialize the graph builder."""
        self.graph = nx.MultiDiGraph()
        
    def build_graph(self, entities: Dict[str, Any], relations: List[Dict[str, Any]]) -> nx.MultiDiGraph:
        """Build a graph from entities and relations.
        
        Args:
            entities: Dictionary of entities
            relations: List of relations
            
        Returns:
            NetworkX graph representing the LS-DYNA model
        """
        # Add entity nodes
        for entity_id, entity_data in entities.items():
            self.graph.add_node(entity_id, **entity_data)
        
        # Add relation edges
        for relation in relations:
            source = relation['source']
            target = relation['target']
            rel_type = relation['type']
            
            if source in self.graph and target in self.graph:
                self.graph.add_edge(source, target, type=rel_type)
        
        return self.graph
    
    def visualize(self, output_file: str = "lsdyna_graph.png", figsize: Tuple[int, int] = (12, 10)):
        """Visualize the graph and save to a file.
        
        Args:
            output_file: Path to save the visualization
            figsize: Figure size as (width, height)
        """
        plt.figure(figsize=figsize)
        
        # Get unique node types for coloring
        node_types = set(nx.get_node_attributes(self.graph, 'type').values())
        color_map = {}
        colors = plt.cm.tab20(range(len(node_types)))
        for i, node_type in enumerate(node_types):
            color_map[node_type] = colors[i]
        
        # Assign node colors based on type
        node_colors = [color_map[self.graph.nodes[node]['type']] for node in self.graph.nodes]
        
        # Get edge types for edge labels
        edge_labels = {(u, v): d['type'] for u, v, d in self.graph.edges(data=True)}
        
        # Position nodes using force-directed layout
        pos = nx.spring_layout(self.graph, k=0.15, iterations=100)
        
        # Draw the graph
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=500, alpha=0.8)
        nx.draw_networkx_edges(self.graph, pos, width=1.0, alpha=0.5, arrows=True)
        nx.draw_networkx_labels(self.graph, pos, font_size=8)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=6)
        
        # Add legend for node types
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=color_map[node_type], markersize=10, label=node_type) 
                          for node_type in node_types]
        plt.legend(handles=legend_handles, loc='upper right')
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graph visualization saved to {output_file}")
    
    def export_to_json(self, output_file: str = "lsdyna_graph.json"):
        """Export the graph to JSON format.
        
        Args:
            output_file: Path to save the JSON file
        """
        # Convert graph to dictionary
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        # Add nodes
        for node_id, node_data in self.graph.nodes(data=True):
            node_entry = {"id": node_id, **node_data}
            graph_data["nodes"].append(node_entry)
        
        # Add edges
        for source, target, edge_data in self.graph.edges(data=True):
            edge_entry = {
                "source": source,
                "target": target,
                **edge_data
            }
            graph_data["edges"].append(edge_entry)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"Graph exported to JSON: {output_file}")


def main():
    """Main function to run the graph builder from command line."""
    parser = argparse.ArgumentParser(description="Build knowledge graph from LS-DYNA LISP expressions")
    parser.add_argument("lisp_file", help="Path to the LISP expressions file")
    parser.add_argument("-o", "--output", help="Output prefix for graph files", default="lsdyna_graph")
    parser.add_argument("-v", "--visualize", action="store_true", help="Generate graph visualization")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.lisp_file):
        print(f"Error: LISP file {args.lisp_file} not found.")
        sys.exit(1)
    
    # Parse LISP file
    lisp_parser = LispParser()
    entities, relations = lisp_parser.parse_file(args.lisp_file)
    
    if not entities:
        print("No entities found in the LISP file.")
        sys.exit(1)
    
    # Build graph
    graph_builder = GraphBuilder()
    graph = graph_builder.build_graph(entities, relations)
    
    print(f"Graph built successfully with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    
    # Export graph
    graph_builder.export_to_json(f"{args.output}.json")
    
    # Visualize if requested
    if args.visualize:
        graph_builder.visualize(f"{args.output}.png")


if __name__ == "__main__":
    main()
