# 46750_assignment_2

This is the code for assigment 2 in 46750

## Table of Contents
- Installation and Setup
- Script Overview
- Usage Examples

## Installation and Setup

Clone or download the repository.

### 1. Create a virtual environment (Python 3.11+)

Using Python directly:
```bash
py -3.11 -m venv .venv
```

Using uv:
```bash
uv venv
```

### 2. Activate your virtual environment

On Windows:
```bash
.venv\Scripts\activate
```

On macOS/Linux:
```bash
source .venv/bin/activate
```

### 3. Install dependencies

Using pip:
```bash
pip install -r requirements.txt
```

Using uv:
```bash
uv sync
```

## Script Overview

## Data

### Sources
The data is synthetic and created via historical data, renewables ninja and the following sources

https://markets.businessinsider.com/commodities/coal-price
https://www.energidataservice.dk/tso-gas/GasDailyBalancingPrice
https://www.renewables.ninja/
https://www.gem.wiki/Coal_power_technologies?utm_source=chatgpt.com
https://energyeducation.ca/encyclopedia/Supercritical_coal_plant?utm_source=chatgpt.com
https://www.idc-online.com/technical_references/pdfs/civil_engineering/Supercritical_coal_fired_power_plant.pdf?utm_source=chatgpt.com
https://iea-etsap.org/E-TechDS/PDF/E02-gas_fired_power-GS-AD-gct_FINAL.pdf?utm_source=chatgpt.com
https://www.acer.europa.eu/sites/default/files/documents/Official_documents/Acts_of_the_Agency/Opinions/Documents/ACERs%20Opinion%2022-2019%20examples%20of%20calculation.pdf?utm_source=chatgpt.com
https://thundersaidenergy.com/downloads/gas-to-power-project-economics/?utm_source=chatgpt.com
https://www.volker-quaschning.de/datserv/CO2-spez/index_e.php?utm_source=chatgpt.com
https://www.gem.wiki/Estimating_carbon_dioxide_emissions_from_gas_plants?utm_source=chatgpt.com
https://icapcarbonaction.com/en/ets-prices

### Assumptions
Missing values are filled by the nearest values, trading on ETS market is possible everyday. 
Conversion rate from USD to EUR 

## Usage Examples
