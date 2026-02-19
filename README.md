# ARC

This repository focuses on solving the Abstraction and Reasoning Corpus (ARC) challenges using Domain-Specific Language (DSL) primitives and LLM-empowered approaches. The project aims to process and solve visual reasoning tasks by building a comprehensive set of modular building blocks that encapsulate concepts like object detection, grid rotation, and pattern recognition. The process of finding a program to solve a given task is treated as a form of program synthesis, utilizing both programmatic composition and Large Language Models to orchestrate solutions.

## Table of Contents

- [Features](#features)
- [Architecture & Codebase Structure](#architecture--codebase-structure)
- [Getting Started](#getting-started)

## Features

- **Domain-Specific Language (DSL):** A robust set of building blocks for grid manipulation, including object detection, rotation, mirroring, edge detection, shifting, and noise generation.
- **Task Visualization:** Utilities to draw and visualize ARC grids and tasks using `drawsvg` and `svgwrite`, supporting SVG, PNG, or PDF formats.
- **LLM Integration:** Sample integrations with Ollama and Llama 3.2 for language model-assisted generation and reasoning via REST APIs.
- **Program Synthesis:** A framework to search over compositions of DSL primitives to discover solution programs for ARC training and test examples.
- **Dataset Augmentation:** Custom scripts and notebooks for augmenting dataset generation and solution synthesis.

## Architecture & Codebase Structure

- `blocks.py`: Core Domain-Specific Language (DSL) containing grid transformation primitives (e.g., `rotate_grid`, `object_detection`, `find_loops`, `detect_edges`).
- `utils.py`: Helper functions for JSON parsing, matrix padding, calculating similarity ratios, and advanced SVG drawing logic (`draw_task_v1`, `draw_grid`).
- `config.py`: Configuration settings, such as ARC grid color mappings (`cmap`).
- `samples/`: Jupyter notebooks demonstrating LLM interaction via API endpoints (`ollama_sample.ipynb`, `rag_sample.ipynb`).
- `solutions/`: Notebooks for synthesizing task solutions by combining DSL primitives (`create_solutions.ipynb`).
- `dataset/`: Custom data generation scripts and generated validation datasets.
- `arc-prize-2024/`: Original ARC training, test, and evaluation challenge datasets along with exploratory notebooks.

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ARC
   ```
2. Install the required dependencies (a virtual environment is recommended):
   ```bash
   pip install numpy drawsvg svgwrite requests jupyter cairosvg
   ```

### Usage
- **Visualize Tasks:** Explore `arc-prize-2024/sample.ipynb` or `solutions/create_solutions.ipynb` to see how ARC tasks are loaded, drawn, and analyzed.
- **LLM Interaction:** Run the notebooks in the `samples/` directory to experiment with local LLM APIs via Ollama.
- **Build Solutions:** Use the primitives in `blocks.py` alongside the `create_solutions.ipynb` notebook to craft and test custom solution sequences for ARC challenges.

---

## Thank You <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Hand%20gestures/Folded%20Hands.png" alt="Folded Hands" width="20" height="20" />

Thank you for checking out **ARC**! We hope this tool makes exploring and solving the Abstraction and Reasoning Corpus (ARC) easier and more efficient. Feel free to fork the repository, try out your own improvements, and contribute. We welcome your feedback and collaboration—your suggestions and pull requests help make this project better for everyone.

**How you can contribute:**
- **Expand the DSL:** Add new grid manipulation primitives or geometric operations to `blocks.py`.
- **Improve Program Synthesis:** Enhance the solution generation logic in the `solutions/` directory to handle more complex abstractions.
- **LLM Prompting:** Optimize prompts or integrate new language models in the `samples/` notebooks.
- Share bug reports, feature requests, or open issues to improve the visualizers and reasoning tools.

We look forward to seeing your ideas and contributions!
