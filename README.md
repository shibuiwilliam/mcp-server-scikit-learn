# mcp-server-scikit-learn: MCP server for Scikit-learn

## Overview

This is a Model Context Protocol server for Scikit-learn, providing a standardized interface for interacting with Scikit-learn models and datasets.

## Features

* Train and evaluate Scikit-learn models
* Handle datasets and data preprocessing
* Model persistence and loading
* Feature engineering and selection
* Model evaluation metrics
* Cross-validation and hyperparameter tuning

## Run this project locally

This project is not yet set up for ephemeral environments (e.g. `uvx` usage). Run this project locally by cloning this repo:

```bash
git clone https://github.com/yourusername/mcp-server-scikit-learn.git
cd mcp-server-scikit-learn
```

You can launch the MCP inspector via npm:

```bash
npx @modelcontextprotocol/inspector uv --directory=src/mcp_server_scikit_learn run mcp-server-scikit-learn
```

Upon launching, the Inspector will display a URL that you can access in your browser to begin debugging.

OR Add this tool as a MCP server:

```json
{
  "scikit-learn": {
    "command": "uv",
    "args": [
      "--directory",
      "/path/to/mcp-server-scikit-learn",
      "run",
      "mcp-server-scikit-learn"
    ]
  }
}
```

## Development

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -e ".[dev]"
```

3. Run tests:
```bash
pytest -s -v tests/
```

## [License](LICENSE)
