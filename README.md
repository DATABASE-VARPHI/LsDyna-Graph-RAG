Key Features

Parsing with pydyna

Leverages the existing pydyna library to parse LS-DYNA input files
Extracts structured data about models (nodes, elements, materials, etc.)


LISP Expression Generation

Converts parsed LS-DYNA data to LISP expressions
Represents entities and relationships in a format suitable for ontology generation


Knowledge Graph Construction

Builds a directed graph representing the LS-DYNA model structure
Identifies relationships between components (e.g., elements reference parts, parts reference materials)


Semantic Search

Converts nodes to textual descriptions
Creates embeddings for efficient similarity search
Uses FAISS for fast retrieval of relevant nodes


Context Extraction

Extracts subgraphs around relevant nodes
Provides neighborhood and path-based context retrieval
Converts subgraphs to textual descriptions for LLM consumption


Integration with LLMs

Designed to work with any LLM through a common interface
Provides prompt templates for effective RAG
