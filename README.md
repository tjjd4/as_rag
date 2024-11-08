# AS-RAG

AS-RAG is a retrieval argumented generation system try to generate responses based on the AS Language Reference Manual, using Hugging Face Transformers and Chroma Vector Database for knowledge retrieval and answer generation.

## Prerequisites

Before installing, ensure you have the following dependencies:

- **Python** 3.12 or higher
- **Poetry** for virtual environment and package management

## File Structure
- `src/`: Contains the core scripts and modules, including data loading (load_data.py) and querying (query_data.py).

- `data/`: Directory for storing the AS Language Reference Manual and other reference data.

## Installation

```bash
git clone https://github.com/your-username/as-rag.git
cd as-rag

# use "poetry" for virtual env and package management
poetry install
```

## Usage
1. Activate virtual enviroment
```bash
poetry shell
```
2. Load data into vector database
```bash
python src/load_data.py
```

3. Ask question in command line
```bash
python src/query_data.py
```

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this software, provided that you include the original copyright notice and this permission notice in all copies or substantial portions of the software.

See the [LICENSE](LICENSE) file for full license details.