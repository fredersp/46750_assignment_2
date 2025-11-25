import pandas as pd
from datetime import datetime
from data_preparation import DataPreparationCSV, DataPreparationJSON 


# Load data files and prepare the dataset for timeseries
data = DataPreparationCSV(datetime(2024,1,1), datetime(2024,12,31),
        coal_file_name="CoalDailyPrices.csv",
        ets_file_name="ETSDailyPrices.csv",
        gas_file_name="GasDailyBalancingPrice.csv",
        wind_file_name="wind_power_prod.csv",
        pv_file_name="pv_power_prod.csv"
    )

df_t = data.build()

params = DataPreparationJSON("appliance_params.json", "storage_params.json")
df_app = params.appliance_data_preparation()
df_stor = params.storage_data_preparation()



