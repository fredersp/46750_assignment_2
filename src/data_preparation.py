from data_loader import load_csv_data, load_json_data
import pandas as pd

class Expando(object):

    pass

class DataPreparationJSON:
    def __init__(self, app_params_file: str, stor_params_file: str):
        self.app_params = load_json_data(app_params_file)
        self.stor_params = load_json_data(stor_params_file)
    
    
    def appliance_data_preparation(self):
        
        data = self.app_params
        df_app = pd.json_normalize(data['DER'])
        
        
        
        return df_app
    
    def storage_data_preparation(self):
        
        data = self.stor_params
        df_stor = pd.json_normalize(data['fuel_storage'])
        
        return df_stor




class DataPreparationCSV:
    
    def __init__(self, start_date: str, end_date: str, coal_file_name: str, ets_file_name: str, gas_file_name: str, 
                 wind_file_name: str, pv_file_name: str):
        
        self.gas_file_name = gas_file_name
        self.coal_file_name = coal_file_name
        self.ets_file_name = ets_file_name
        self.wind_file_name = wind_file_name
        self.pv_file_name = pv_file_name
        self.start_date = start_date
        self.end_date = end_date
        
        self.build()
    

    def prepare_coal_data(self):
        
        data = load_csv_data(self.coal_file_name)
        
        # Extract relevant columns and filter by date range
        coal_prices = data['Close']
        dates = data['Date']
        dates = pd.to_datetime(dates, format='%m/%d/%y')
        mask = (dates >= self.start_date) & (dates <= self.end_date)
        dates = dates.loc[mask].reset_index(drop=True)
        coal_prices = coal_prices.loc[mask].reset_index(drop=True)
        
        # Convert coal price to EUR/KWh from USD/tonne
        ird_to_eur = 1.15  # EUR/USD
        tonnes_to_kwh = 0.00012283503255128   # KWh/tonne coal
        coal_prices = coal_prices * ird_to_eur * tonnes_to_kwh
        
        coal = pd.Series(coal_prices.values, index=dates, name="Coal_Price")
        
        return coal
    
    def prepare_ets_data(self):
        
        # Load ETS data from CSV
        data = load_csv_data(self.ets_file_name, header=1)
        
        # Extract relevant columns and filter by date range
        ets_prices = data['Primary Market']
        dates = data['Date']
        dates = pd.to_datetime(dates, format='%Y-%m-%d')
        mask = (dates >= self.start_date) & (dates <= self.end_date)
        dates = dates.loc[mask].reset_index(drop=True)
        ets_prices = ets_prices.loc[mask].reset_index(drop=True)
        
        # Create a Pandas Series for ETS prices
        ets = pd.Series(ets_prices.values, index=dates, name="ETS_Price")
        
        # convert to from tonne to EUR/kgCO2eq
        ets = ets * 0.001
    
        return ets
    
    def prepare_gas_data(self):
        data = load_csv_data(self.gas_file_name, sep=';', decimal=',')
        
        # Extract relevant columns and filter by date range
        gas_prices = data['EEXSpotIndexEUR_MWh']
        dates = data['GasDay']
        dates = pd.to_datetime(dates, format= '%Y-%m-%d %H:%M:%S')
        mask = (dates >= self.start_date) & (dates <= self.end_date)
        dates = dates.loc[mask].reset_index(drop=True)
        gas_prices = gas_prices.loc[mask].reset_index(drop=True)
        
        gas = pd.Series(gas_prices.values, index=dates, name="Gas_Price")
        
        # Convert to EUR/KWh
        gas = gas * 0.001
        
        return gas
    
    
    def prepare_wind_data(self):
        data = load_csv_data(self.wind_file_name, skiprow=3)
        data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M')
        data['electricity'] = pd.to_numeric(data['electricity'], errors='coerce')

        daily_wind = data.set_index('time')['electricity'].resample('D').sum()

        # shift 2019 dates to 2024
        daily_wind.index = daily_wind.index.map(lambda dt: dt.replace(year=2024))

        dates = pd.date_range(self.start_date, self.end_date, freq='D')
        daily_wind = daily_wind.reindex(dates)

        # fill leap-day gap (2019 had no Feb 29)
        if pd.Timestamp('2024-02-29') in daily_wind.index:
            daily_wind.loc['2024-02-29'] = daily_wind.loc[['2024-02-28', '2024-03-01']].mean()

        return daily_wind*0.1

    def prepare_pv_data(self):
        data = load_csv_data(self.pv_file_name, skiprow=3)
        data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M')
        data['electricity'] = pd.to_numeric(data['electricity'], errors='coerce')

        daily_pv = data.set_index('time')['electricity'].resample('D').sum()

        # shift 2019 dates to 2024
        daily_pv.index = daily_pv.index.map(lambda dt: dt.replace(year=2024))

        dates = pd.date_range(self.start_date, self.end_date, freq='D')
        daily_pv = daily_pv.reindex(dates)

        # fill leap-day gap (2019 had no Feb 29)
        if pd.Timestamp('2024-02-29') in daily_pv.index:
            daily_pv.loc['2024-02-29'] = daily_pv.loc[['2024-02-28', '2024-03-01']].mean()

        return daily_pv*0.5        
        
    
    
    def build(self):
        # Create a date range for the entire period
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        # Coal prices
        coal_prices = self.prepare_coal_data().reindex(dates, method='nearest')
        
        # ETS prices
        ets_prices = self.prepare_ets_data()
        
        # Make sure ETS prices cover all dates by reindexing and filling missing values
        ets_prices = ets_prices.interpolate(method='time').ffill().bfill()
        ets_prices = ets_prices.reindex(dates, method='nearest')
        
        # Gas prices
        gas_prices = self.prepare_gas_data().reindex(dates, method='nearest')
        
        # wind data
        wind_prod = self.prepare_wind_data()
        
        # PV data
        pv_prod = self.prepare_pv_data()
        
        
        df = pd.DataFrame({
            'Date': dates,
            'Coal_Price[EUR/KWh]': coal_prices,
            'ETS_Price[EUR/kgCO2eq]': ets_prices,
            'Gas_Price[EUR/KWh]': gas_prices,
            'Wind_Prod[KWh]': wind_prod,
            'PV_Prod[KWh]': pv_prod       
        })
            
        
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df

