#!/usr/bin/env python3
"""
LS-DYNA Graph RAG - Main Module
------------------------------
This module implements a graph-based RAG system for LS-DYNA models.
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Any, Tuple, Optional
import logging
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import faiss
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

# Local imports
try:
    from lsdyna_parser import LSDynaParser
    from lsdyna_graph_builder import LispParser, GraphBuilder
except ImportError:
    print("Error: Required local modules not found. Make sure lsdyna_parser.py and lsdyna_graph_builder.py are in your path.")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class GraphEmbedding:
    """Class to create and manage embeddings for graph nodes."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.embeddings = {}
        self.node_texts = {}
        self.index = None
        
    def node_to_text(self, node_id: str, node_data: Dict[str, Any]) -> str:
        """Convert a graph node to a textual representation.
        
        Args:
            node_id: ID of the node
            node_data: Data associated with the node
            
        Returns:
            String representation of the node
        """
        text_parts = [f"ID: {node_id}"]
        text_parts.append(f"Type: {node_data.get('type', 'Unknown')}")
        
        if 'properties' in node_data:
            props = []
            for key, value in node_data['properties'].items():
                props.append(f"{key}: {value}")
            
            if props:
                text_parts.append("Properties: " + ", ".join(props))
        
        return " | ".join(text_parts)
    
    def create_embeddings(self, graph: nx.MultiDiGraph):
        """Create embeddings for all nodes in the graph.
        
        Args:
            graph: NetworkX graph with LS-DYNA entities
        """
        logger.info("Creating embeddings for graph nodes...")
        
        # Convert nodes to text
        for node_id, node_data in graph.nodes(data=True):
            self.node_texts[node_id] = self.node_to_text(node_id, node_data)
        
        # Generate embeddings
        texts = list(self.node_texts.values())
        embeddings = self.model.encode(texts)
        
        # Store embeddings
        for idx, node_id in enumerate(self.node_texts.keys()):
            self.embeddings[node_id] = embeddings[idx]
        
        # Build FAISS index for efficient similarity search
        self._build_index()
        
        logger.info(f"Created embeddings for {len(self.embeddings)} nodes.")
    
    def _build_index(self):
        """Build a FAISS index for the embeddings."""
        if not self.embeddings:
            logger.warning("No embeddings to build index.")
            return
        
        # Get dimension from first embedding
        dim = next(iter(self.embeddings.values())).shape[0]
        
        # Initialize index
        self.index = faiss.IndexFlatL2(dim)
        
        # Add embeddings to index
        node_ids = list(self.embeddings.keys())
        embeddings_matrix = np.vstack([self.embeddings[nid] for nid in node_ids])
        self.index.add(embeddings_matrix)
        
        # Store node IDs for lookup
        self.indexed_node_ids = node_ids
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, str]]:
        """Search for graph nodes similar to query.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of tuples (node_id, similarity_score, node_text)
        """
        if not self.index:
            logger.error("Index not built. Call create_embeddings first.")
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search index
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.indexed_node_ids):  # Check if index is valid
                node_id = self.indexed_node_ids[idx]
                distance = distances[0][i]
                similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity score
                node_text = self.node_texts[node_id]
                results.append((node_id, similarity, node_text))
        
        return results


class SubgraphExtractor:
    """Class to extract relevant subgraphs for context."""
    
    def __init__(self, graph: nx.MultiDiGraph):
        """Initialize with the full graph.
        
        Args:
            graph: NetworkX graph representing the LS-DYNA model
        """
        self.graph = graph
    
    def extract_neighborhood(self, node_ids: List[str], depth: int = 1) -> nx.MultiDiGraph:
        """Extract a subgraph containing the neighborhood of specified nodes.
        
        Args:
            node_ids: List of node IDs to use as centers
            depth: Depth of neighborhood to include
            
        Returns:
            NetworkX subgraph
        """
        # Start with seed nodes
        nodes_to_include = set(node_ids)
        
        # Expand neighborhood based on depth
        current_nodes = set(node_ids)
        for _ in range(depth):
            next_level = set()
            for node in current_nodes:
                if node in self.graph:
                    neighbors = set(self.graph.predecessors(node)) | set(self.graph.successors(node))
                    next_level.update(neighbors)
            
            nodes_to_include.update(next_level)
            current_nodes = next_level
        
        # Create subgraph
        return self.graph.subgraph(nodes_to_include).copy()
    
    def extract_paths(self, source_nodes: List[str], target_nodes: List[str], 
                     max_length: int = 3) -> nx.MultiDiGraph:
        """Extract paths between source and target nodes.
        
        Args:
            source_nodes: List of source node IDs
            target_nodes: List of target node IDs
            max_length: Maximum path length
            
        Returns:
            NetworkX subgraph containing the paths
        """
        nodes_in_paths = set()
        
        # Find all paths between sources and targets
        for source in source_nodes:
            for target in target_nodes:
                try:
                    # Use simple paths with limit on length
                    for path in nx.all_simple_paths(self.graph, source, target, cutoff=max_length):
                        nodes_in_paths.update(path)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # Handle case when no path exists or nodes not found
                    continue
        
        # Create subgraph from nodes in paths
        if nodes_in_paths:
            return self.graph.subgraph(nodes_in_paths).copy()
        else:
            # Fallback to neighborhood if no paths found
            return self.extract_neighborhood(source_nodes + target_nodes, depth=1)


class GraphToText:
    """Class to convert graph structures to textual descriptions."""
    
    @staticmethod
    def node_to_text(node_id: str, node_data: Dict[str, Any]) -> str:
        """Convert a node to a textual description.
        
        Args:
            node_id: ID of the node
            node_data: Data associated with the node
            
        Returns:
            Text description of the node
        """
        lines = [f"Node: {node_id}"]
        lines.append(f"Type: {node_data.get('type', 'Unknown')}")
        
        if 'properties' in node_data:
            lines.append("Properties:")
            for key, value in node_data['properties'].items():
                lines.append(f"  - {key}: {value}")
        
        return "\n".join(lines)
    
    @staticmethod
    def edge_to_text(source: str, target: str, edge_data: Dict[str, Any]) -> str:
        """Convert an edge to a textual description.
        
        Args:
            source: Source node ID
            target: Target node ID
            edge_data: Data associated with the edge
            
        Returns:
            Text description of the edge
        """
        edge_type = edge_data.get('type', 'RELATED_TO')
        return f"Relationship: {source} --[{edge_type}]--> {target}"
    
    @staticmethod
    def graph_to_text(graph: nx.MultiDiGraph) -> str:
        """Convert a graph to a textual description.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Text description of the graph
        """
        text_parts = [f"LS-DYNA Graph Summary:",
                     f"- Nodes: {graph.number_of_nodes()}",
                     f"- Relationships: {graph.number_of_edges()}\n"]
        
        # Add nodes
        text_parts.append("Entities:")
        for node_id, node_data in graph.nodes(data=True):
            text_parts.append(GraphToText.node_to_text(node_id, node_data))
            text_parts.append("")  # Empty line for separation
        
        # Add edges
        text_parts.append("Relationships:")
        for source, target, edge_data in graph.edges(data=True):
            text_parts.append(GraphToText.edge_to_text(source, target, edge_data))
        
        return "\n".join(text_parts)


class LSDynaGraphRAG:
    """Main class for the LS-DYNA Graph RAG system."""
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize the LS-DYNA Graph RAG system.
        
        Args:
            llm_config: Configuration for the language model (optional)
        """
        self.graph = None
        self.graph_embeddings = GraphEmbedding()
        self.subgraph_extractor = None
        self.llm_config = llm_config or {}
        self.parser = None
        
    def process_input_file(self, input_file: str, lisp_file: str = "output.lisp") -> nx.MultiDiGraph:
        """Process an LS-DYNA input file to build the graph.
        
        Args:
            input_file: Path to the LS-DYNA input file
            lisp_file: Path to save intermediate LISP expressions
            
        Returns:
            NetworkX graph representing the LS-DYNA model
        """
        logger.info(f"Processing LS-DYNA input file: {input_file}")
        
        # Step 1: Parse LS-DYNA input file with pydyna
        self.parser = LSDynaParser(verbose=True)
        parsed_data = self.parser.parse_file(input_file)
        
        if not parsed_data:
            logger.error("Failed to parse input file.")
            return None
        
        # Save LISP expressions
        self.parser.save_lisp(lisp_file)
        logger.info(f"Generated LISP expressions: {lisp_file}")
        
        # Step 2: Build graph from LISP expressions
        lisp_parser = LispParser()
        entities, relations = lisp_parser.parse_file(lisp_file)
        
        if not entities:
            logger.error("No entities found in LISP expressions.")
            return None
        
        # Build graph
        graph_builder = GraphBuilder()
        self.graph = graph_builder.build_graph(entities, relations)
        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        
        # Initialize subgraph extractor
        self.subgraph_extractor = SubgraphExtractor(self.graph)
        
        # Create embeddings
        self.graph_embeddings.create_embeddings(self.graph)
        
        return self.graph
    
    def load_graph(self, graph_file: str) -> nx.MultiDiGraph:
        """Load a graph from a JSON file.
        
        Args:
            graph_file: Path to the graph JSON file
            
        Returns:
            NetworkX graph
        """
        logger.info(f"Loading graph from: {graph_file}")
        
        try:
            with open(graph_file, 'r') as f:
                graph_data = json.load(f)
            
            # Create new graph
            self.graph = nx.MultiDiGraph()
            
            # Add nodes
            for node in graph_data.get("nodes", []):
                node_id = node.pop("id")
                self.graph.add_node(node_id, **node)
            
            # Add edges
            for edge in graph_data.get("edges", []):
                source = edge.pop("source")
                target = edge.pop("target")
                self.graph.add_edge(source, target, **edge)
            
            # Initialize subgraph extractor and embeddings
            self.subgraph_extractor = SubgraphExtractor(self.graph)
            self.graph_embeddings.create_embeddings(self.graph)
            
            logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
            return self.graph
            
        except Exception as e:
            logger.error(f"Error loading graph: {str(e)}")
            return None
    
    def query(self, query_text: str, k: int = 5, context_depth: int = 1) -> Dict[str, Any]:
        """Process a query and retrieve relevant context from the graph.
        
        Args:
            query_text: Query text
            k: Number of similar nodes to retrieve
            context_depth: Depth of context to include
            
        Returns:
            Dictionary with query results and augmented context
        """
        if not self.graph or not self.graph_embeddings.index:
            logger.error("Graph or embeddings not initialized. Process an input file first.")
            return {"error": "Graph not initialized"}
        
        # Step 1: Find relevant nodes
        similar_nodes = self.graph_embeddings.search(query_text, k=k)
        
        if not similar_nodes:
            logger.warning("No similar nodes found for query.")
            return {"error": "No relevant information found"}
        
        # Extract node IDs and texts
        relevant_node_ids = [node_id for node_id, _, _ in similar_nodes]
        relevant_node_texts = [text for _, _, text in similar_nodes]
        
        # Step 2: Extract relevant subgraph
        context_graph = self.subgraph_extractor.extract_neighborhood(relevant_node_ids, depth=context_depth)
        
        # Step 3: Convert graph to text
        context_text = GraphToText.graph_to_text(context_graph)
        
        # Step 4: Prepare result
        result = {
            "query": query_text,
            "relevant_nodes": [{"id": node_id, "similarity": sim, "text": text} 
                             for node_id, sim, text in similar_nodes],
            "context_graph": {
                "nodes": context_graph.number_of_nodes(),
                "edges": context_graph.number_of_edges()
            },
            "context_text": context_text
        }
        
        return result
    
    def generate_response(self, query_text: str, llm_client=None) -> Dict[str, Any]:
        """Generate a response to a query using RAG with the graph context.
        
        Args:
            query_text: Query text
            llm_client: LLM client (optional)
            
        Returns:
            Dictionary with query results and generated response
        """
        # Get context from graph
        query_result = self.query(query_text)
        
        if "error" in query_result:
            return query_result
        
        # Create prompt with context
        prompt_template = """
        You are an expert assistant for LS-DYNA finite element analysis software.
        Below is information about an LS-DYNA model extracted from a knowledge graph:
        
        {context}
        
        Based on this information, please answer the following question:
        {query}
        
        Provide a detailed and technical response that directly addresses the question.
        """
        
        prompt = PromptTemplate.from_template(prompt_template)
        
        # If LLM client provided, generate response
        if llm_client:
            chain = (
                {"context": lambda x: x["context_text"], "query": lambda x: x["query"]}
                | prompt
                | llm_client
                | StrOutputParser()
            )
            
            response = chain.invoke(query_result)
            query_result["response"] = response
        
        return query_result


def main():
    """Main function to run the LS-DYNA Graph RAG system."""
    parser = argparse.ArgumentParser(description="LS-DYNA Graph RAG System")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process an LS-DYNA input file")
    process_parser.add_argument("input_file", help="Path to the LS-DYNA input file")
    process_parser.add_argument("-l", "--lisp", help="Path to save LISP expressions", default="output.lisp")
    process_parser.add_argument("-o", "--output", help="Path to save the graph", default="lsdyna_graph.json")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the graph")
    query_parser.add_argument("graph_file", help="Path to the graph JSON file")
    query_parser.add_argument("query", help="Query text")
    query_parser.add_argument("-k", "--top_k", type=int, default=5, help="Number of similar nodes to retrieve")
    query_parser.add_argument("-d", "--depth", type=int, default=1, help="Context depth")
    
    args = parser.parse_args()
    
    # Initialize the system
    rag_system = LSDynaGraphRAG()
    
    if args.command == "process":
        # Process input file
        if not os.path.exists(args.input_file):
            logger.error(f"Input file {args.input_file} not found.")
            sys.exit(1)
        
        graph = rag_system.process_input_file(args.input_file, args.lisp)
        
        if graph:
            # Save graph
            graph_builder = GraphBuilder()
            graph_builder.graph = graph
            graph_builder.export_to_json(args.output)
            logger.info(f"Graph saved to {args.output}")
        
    elif args.command == "query":
        # Query the graph
        if not os.path.exists(args.graph_file):
            logger.error(f"Graph file {args.graph_file} not found.")
            sys.exit(1)
        
        # Load graph
        rag_system.load_graph(args.graph_file)
        
        # Process query
        result = rag_system.query(args.query, k=args.top_k, context_depth=args.depth)
        
        if "error" in result:
            logger.error(result["error"])
        else:
            # Print query results
            print("\n" + "="*50)
            print(f"QUERY: {result['query']}")
            print("="*50)
            
            print("\nRELEVANT ENTITIES:")
            for idx, node in enumerate(result['relevant_nodes']):
                print(f"{idx+1}. {node['id']} (Score: {node['similarity']:.3f})")
                print(f"   {node['text']}")
            
            print("\nCONTEXT SUMMARY:")
            print(f"Extracted subgraph with {result['context_graph']['nodes']} nodes and {result['context_graph']['edges']} edges.")
            
            print("\nCONTEXT DETAILS:")
            print(result['context_text'][:500] + "..." if len(result['context_text']) > 500 else result['context_text'])
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()