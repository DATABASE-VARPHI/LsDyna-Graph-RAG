#!/usr/bin/env python3
"""
LS-DYNA Graph RAG - Example Usage
-------------------------------
This script demonstrates how to use the LS-DYNA Graph RAG system.
"""

import os
import sys
import logging
from pathlib import Path
import json

# Import the LS-DYNA Graph RAG modules
try:
    from lsdyna_parser import LSDynaParser
    from lsdyna_graph_builder import GraphBuilder, LispParser
    from lsdyna_rag import LSDynaGraphRAG
except ImportError:
    print("Error: Required modules not found. Make sure all modules are in your Python path.")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample LS-DYNA input content for demonstration
SAMPLE_LSDYNA_INPUT = """*KEYWORD
*TITLE
LS-DYNA Example
*PART
$#     pid     secid       mid     eosid      hgid      grav    adpopt      tmid
         1         1         1         0         0         0         0         0
*SECTION_SHELL
$#   secid    elform      shrf       nip     propt   qr/irid     icomp     setyp
         1         2       1.0         2       1.0         0         0         1
$#      t1        t2        t3        t4      nloc     marea      idof    edgset
       1.0       1.0       1.0       1.0       0.0       0.0       0.0         0
*MAT_ELASTIC
$#     mid        ro         e        pr        da        db  not used        
         1    7.8E-9   2.10E+5      0.30       0.0       0.0       0.0
*ELEMENT_SHELL
$#   eid     pid      n1      n2      n3      n4
       1       1       1       2       3       4
       2       1       5       6       7       8
*NODE
$#   nid               x               y               z      tc      rc  
       1             0.0             0.0             0.0       0       0
       2            10.0             0.0             0.0       0       0
       3            10.0            10.0             0.0       0       0
       4             0.0            10.0             0.0       0       0
       5             0.0             0.0            10.0       0       0
       6            10.0             0.0            10.0       0       0
       7            10.0            10.0            10.0       0       0
       8             0.0            10.0            10.0       0       0
*END
"""

def create_sample_input_file():
    """Create a sample LS-DYNA input file."""
    with open("sample_lsdyna.k", "w") as f:
        f.write(SAMPLE_LSDYNA_INPUT)
    logger.info("Created sample LS-DYNA input file: sample_lsdyna.k")
    return "sample_lsdyna.k"

def process_example():
    """Process the sample LS-DYNA file and create a graph."""
    # Create sample input file
    input_file = create_sample_input_file()
    
    # Initialize the LS-DYNA Graph RAG system
    rag_system = LSDynaGraphRAG()
    
    # Process the input file
    logger.info("Processing LS-DYNA input file...")
    graph = rag_system.process_input_file(input_file, lisp_file="sample_output.lisp")
    
    if graph:
        logger.info(f"Successfully built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        # Visualize the graph
        graph_builder = GraphBuilder()
        graph_builder.graph = graph
        graph_builder.visualize("sample_graph.png")
        graph_builder.export_to_json("sample_graph.json")
        
        return graph
    else:
        logger.error("Failed to build graph from sample input file")
        return None

def query_example(graph_file="sample_graph.json"):
    """Run example queries against the graph."""
    # Check if graph file exists
    if not os.path.exists(graph_file):
        logger.error(f"Graph file {graph_file} not found. Run process_example() first.")
        return None
    
    # Initialize the LS-DYNA Graph RAG system
    rag_system = LSDynaGraphRAG()
    
    # Load the graph
    graph = rag_system.load_graph(graph_file)
    
    if not graph:
        logger.error("Failed to load graph")
        return None
    
    # Example queries
    queries = [
        "What material properties are used in the model?",
        "Tell me about the shell elements in the model",
        "How many nodes are in the model?",
        "What is the thickness of the shell elements?"
    ]
    
    # Process each query
    results = []
    for query in queries:
        logger.info(f"Processing query: {query}")
        result = rag_system.query(query)
        results.append(result)
        
        # Print results
        print("\n" + "="*50)
        print(f"QUERY: {query}")
        print("="*50)
        
        print("\nRELEVANT ENTITIES:")
        for idx, node in enumerate(result['relevant_nodes']):
            print(f"{idx+1}. {node['id']} (Score: {node['similarity']:.3f})")
            print(f"   {node['text']}")
        
        print("\n")
    
    return results

def generate_responses_example(graph_file="sample_graph.json"):
    """Example of generating responses with a mock LLM."""
    # Check if graph file exists
    if not os.path.exists(graph_file):
        logger.error(f"Graph file {graph_file} not found. Run process_example() first.")
        return None
    
    # Initialize the system
    rag_system = LSDynaGraphRAG()
    
    # Load the graph
    graph = rag_system.load_graph(graph_file)
    
    if not graph:
        logger.error("Failed to load graph")
        return None
    
    # Create a mock LLM for demonstration
    class MockLLM:
        def invoke(self, prompt):
            # This is just a placeholder that would normally be replaced with a real LLM API call
            query = prompt.split("question:")[-1].strip()
            context = prompt.split("Below is information")[1].split("Based on this information")[0].strip()
            
            response = f"This is a demonstration response for query: '{query}'\n\n"
            response += "Based on the LS-DYNA model information, I can tell you that:\n\n"
            
            # Add some context-aware mock response
            if "material" in query.lower():
                response += "The model uses elastic material (MAT_ELASTIC) with ID 1, density of 7.8E-9, "
                response += "Young's modulus of 2.1E+5, and Poisson's ratio of 0.3."
            elif "shell" in query.lower():
                response += "The model contains shell elements with section ID 1, using formulation 2, "
                response += "with a thickness of 1.0 units. There are 2 shell elements defined with IDs 1 and 2."
            elif "node" in query.lower():
                response += "The model contains 8 nodes forming the corners of a 10x10x10 cube."
            else:
                response += "The model is a simple cube made of 2 shell elements with 8 nodes, "
                response += "using elastic material properties."
                
            return response
    
    # Example queries
    queries = [
        "What material properties are used in the model?",
        "Tell me about the shell elements in the model",
        "How many nodes are in the model?"
    ]
    
    # Process each query with mock LLM
    results = []
    mock_llm = MockLLM()
    
    for query in queries:
        logger.info(f"Processing query with LLM: {query}")
        result = rag_system.generate_response(query, llm_client=mock_llm)
        results.append(result)
        
        # Print results
        print("\n" + "="*50)
        print(f"QUERY: {query}")
        print("="*50)
        
        print("\nRAG CONTEXT:")
        print(f"Using {result['context_graph']['nodes']} nodes and {result['context_graph']['edges']} edges")
        
        print("\nGENERATED RESPONSE:")
        print(result.get('response', 'No response generated'))
        
        print("\n")
    
    return results

def advanced_example():
    """Run a more advanced example using the Graph RAG system."""
    # Initialize the system
    rag_system = LSDynaGraphRAG()
    
    # First check if we already have the graph
    if os.path.exists("sample_graph.json"):
        logger.info("Loading existing graph...")
        graph = rag_system.load_graph("sample_graph.json")
    else:
        # Process the sample file
        logger.info("Creating new graph from sample file...")
        graph = process_example()
    
    if not graph:
        logger.error("Failed to get graph")
        return
    
    # Example of path-based subgraph extraction
    logger.info("Demonstrating path-based subgraph extraction...")
    
    # Get some node IDs from the graph for demonstration
    node_ids = list(graph.nodes())
    if len(node_ids) < 2:
        logger.error("Not enough nodes in graph for path demonstration")
        return
    
    # Extract a subgraph containing paths
    subgraph_extractor = rag_system.subgraph_extractor
    source_nodes = node_ids[:1]  # Take first node as source
    target_nodes = node_ids[-1:]  # Take last node as target
    
    # Extract paths subgraph
    paths_subgraph = subgraph_extractor.extract_paths(source_nodes, target_nodes)
    
    logger.info(f"Paths subgraph contains {paths_subgraph.number_of_nodes()} nodes and "
               f"{paths_subgraph.number_of_edges()} edges")
    
    # Convert to text
    from lsdyna_rag import GraphToText
    subgraph_text = GraphToText.graph_to_text(paths_subgraph)
    
    print("\nPATH SUBGRAPH TEXT:")
    print(subgraph_text[:500] + "..." if len(subgraph_text) > 500 else subgraph_text)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("LS-DYNA GRAPH RAG EXAMPLES")
    print("="*50)
    
    # Run examples
    process_example()
    print("\n\n")
    query_example()
    print("\n\n")
    generate_responses_example()
    print("\n\n")
    advanced_example()
