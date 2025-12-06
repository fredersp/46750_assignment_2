# 46750 – Assignment 2  
**Optimization in Modern Power Systems (Fall 2025)**  
**Group 4**

**Group Members**  
- Christian Witt (s203667)  
- Frederik Heide Tvede (s201163)  
- Frederik Springer Krehan (s203684)  
- Martha Marie Halkjær Kofod (s203703)

> **Note:** Some scripts require significant computation time. Comment out unnecessary analyses if only a single experiment is needed.

---

## Table of Contents
- Installation and Setup
- Script Overview
- Data
- Usage Examples

---

## Installation and Setup

Clone or download the repository to your local machine.

### 1. Create a virtual environment (Python 3.11+)

Using Python:
```bash
py -3.11 -m venv .venv
```

Using `uv`:
```bash
uv venv
```

### 2. Activate the virtual environment

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

Using pip:
```bash
pip install -r requirements.txt
```

Using `uv`:
```bash
uv sync
```

---

## Script Overview

### `data_loader.py`
Provides a class for loading CSV and JSON files.

### `data_preparation.py`
Uses `data_loader` to prepare raw data and returns cleaned DataFrames for use in the optimization models.

### `deterministic_opt_model.py`
Defines:
- `InputData` class for structured model input  
- `DeterministicModel` class implementing the optimization problem (variables, constraints, objective)

Includes two model variants:  
1. Without storage costs  
2. With storage costs  

### `main_deterministic.py`
Loads data, runs the deterministic model, performs initial data analysis, and executes several experiments.

### `stochastic_opt_model.py`
Defines:
- `InputData` class  
- `StochasticModel` class with formulation and result handling  

Includes two variants:  
1. Without risk aversion  
2. With risk aversion  

### `main_stochastic.py`
Loads data, runs the stochastic model, presents results, and performs four additional experiments.

### `multi_stage_stochastic_opt_model.py`
Contains:
- `InputData` class  
- `FirstStageClass` and `SecondStageClass` for the multi-stage model  

Includes:  
1. A model without risk aversion  
2. A model with risk aversion  

### `main_multi_stochastic.py`
Loads data and runs the multi-stage stochastic model. No additional experiments are included.

### `plotter.py`
Helper module with plotting functions.

### `results.py`
Collects and visualizes key model results.

---

## Data
The dataset consists of real time-series data and approximated values based on research and ChatGPT-generated assumptions.

### Data Sources

#### ETS Prices  
https://icapcarbonaction.com/en/ets-prices  

#### Gas Prices  
https://www.energidataservice.dk/tso-gas/GasDailyBalancingPrice  

#### Coal Prices  
https://markets.businessinsider.com/commodities/coal-price  

#### PV and Wind Data  
https://www.renewables.ninja/

---

## Usage Examples

To run the deterministic model:

```bash
python main_deterministic.py
```

This will generate multiple plots and result summaries automatically.
