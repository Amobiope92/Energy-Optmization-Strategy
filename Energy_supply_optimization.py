#!/usr/bin/env python
# coding: utf-8

# In[5]:

# streamlit run "C:\Users\YADTECH\Documents\Data_science\SAIL Exercises\SAIL CAPSTONE PROJECT DOCUMENTS\Optimization of Energy Supply Management\Energy_supply_optimization.py"

import streamlit as st
import pandas as pd
import pulp as lp
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Function to optimize energy strategy
def optimize_energy_strategy(merged_data):
    # Constants
    time_intervals = len(merged_data)  # Number of time intervals in the dataset.
    battery_capacity_kwh = 100  # Total battery capacity in kWh.
    battery_threshold = 20  # Minimum level for the battery, below which it shouldn't drop.
    initial_battery_level = 50  # Starting level of the battery in kWh.
    time_interval = 60  # Duration of each time period (60 minutes).

    # Cost parameters
    cost_grid = 0.15  # $ per kWh from the grid
    cost_solar = 0.05  # $ per kWh from solar
    cost_diesel = 0.20  # $ per kWh from diesel

    predicted_solar = merged_data['predicted_solar_output'].values
    demand = merged_data['Total Energy(kWh)'].values
    grid_outage = merged_data['grid outage plan'].values

    # Initialize the optimization problem
    prob = lp.LpProblem("Minimize_Energy_Cost", lp.LpMinimize)

    # Define decision variables for each interval
    usage_grid = [lp.LpVariable(f"usage_grid_{t}", 0, demand[t]) for t in range(time_intervals)]
    usage_diesel = [lp.LpVariable(f"usage_diesel_{t}", 0, demand[t]) for t in range(time_intervals)]
    usage_solar = [lp.LpVariable(f"usage_solar_{t}", 0, predicted_solar[t]) for t in range(time_intervals)]

    # Objective: Minimize total energy costs
    prob += lp.lpSum([cost_grid * usage_grid[t] + cost_diesel * usage_diesel[t] for t in range(time_intervals)])

    # Add constraints
    for t in range(time_intervals):
        prob += usage_grid[t] + usage_diesel[t] + usage_solar[t] == demand[t]  # Ensure demand is met
        if grid_outage[t] == 1:
            prob += usage_grid[t] == 0  # Prevent grid usage during outages

    # Solve the optimization problem
    prob.solve()

    # Generate strategy based on optimization results
    strategy = []
    battery_level = initial_battery_level
    site_names = list(merged_data['Site Name'].unique())
    time_intervals = len(merged_data) // len(site_names)

    for site in site_names:
        for t in range(time_intervals):
            battery_level = max(battery_threshold, battery_level - demand[t] + predicted_solar[t])
            strategy.append({
                "site_name": site,
                "time": t + 1,
                "grid": usage_grid[t].varValue > 0,
                "solar": usage_solar[t].varValue > 0,
                "diesel": usage_diesel[t].varValue > 0,
                "battery_level": battery_level
            })

    strategy_df = pd.DataFrame(strategy)
    return strategy_df

# Streamlit app
st.title("Energy Optimization Strategy")

uploaded_energy = st.file_uploader("Upload Energy Consumption Dataset", type=["csv"])
uploaded_site = st.file_uploader("Upload Site Information Dataset", type=["csv"])
uploaded_solar = st.file_uploader("Upload Solar Information Dataset", type=["csv"])

if uploaded_energy and uploaded_site and uploaded_solar:
    # Load the datasets
    energy_consumption = pd.read_csv(uploaded_energy)
    site_info = pd.read_csv(uploaded_site)
    solar_info = pd.read_csv(uploaded_solar)

    # Preprocess and merge datasets
    merged_data = pd.merge(energy_consumption, solar_info, on=['Site Name', 'Day', 'Hour'])
    site_info.rename(columns={'Site Id': 'Site Name'}, inplace=True)
    merged_data = pd.merge(merged_data, site_info, on='Site Name')
    merged_data['Hour'] = pd.to_datetime(merged_data['Hour'], errors='coerce')
    merged_data['Hour'] = merged_data['Hour'].dt.hour

    # Train model to predict solar output
    X = merged_data[['Solar Zenith Angle', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Relative Humidity', 'Day', 'Hour']]
    y = merged_data['Energy Output(kWh)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    solar_model = RandomForestRegressor(n_estimators=100, random_state=42)
    solar_model.fit(X_train, y_train)
    merged_data['predicted_solar_output'] = solar_model.predict(X)

    # Optimize energy strategy
    strategy_df = optimize_energy_strategy(merged_data)

    # Display the strategy on the interface
    st.subheader("Optimized Energy Strategy")
    st.dataframe(strategy_df)

    # Optional: Save to CSV (commented out as per requirement)
    # strategy_df.to_csv('strategy.csv', index=False)
else:
    st.warning("Please upload all three datasets to proceed.")




# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# import pulp as lp

# # Streamlit App Title
# st.title('Energy Cost Minimization and Supply Management Optimization')

# # Sidebar for uploading data files
# st.sidebar.header('Upload CSV Files')
# energy_file = st.sidebar.file_uploader('Upload Energy Consumption CSV', type='csv')
# site_info_file = st.sidebar.file_uploader('Upload Site Information CSV', type='csv')
# solar_info_file = st.sidebar.file_uploader('Upload Solar Power CSV', type='csv')

# if energy_file and site_info_file and solar_info_file:
#     # Load datasets
#     energy_consumption = pd.read_csv(energy_file)
#     site_info = pd.read_csv(site_info_file)
#     solar_info = pd.read_csv(solar_info_file)

#     # Display basic dataset information
#     st.subheader('Energy Consumption Data Overview')
#     st.write(energy_consumption.head())
#     st.write(f"Shape: {energy_consumption.shape}")
#     st.write(energy_consumption.describe())

#     st.subheader('Site Information Overview')
#     st.write(site_info.head())
#     st.write(f"Shape: {site_info.shape}")
#     st.write(site_info.describe())

#     st.subheader('Solar Power and Weather Condition Overview')
#     st.write(solar_info.head())
#     st.write(f"Shape: {solar_info.shape}")
#     st.write(solar_info.describe())

#     # Data merging
#     site_info.rename(columns={'Site Id': 'Site Name'}, inplace=True)
#     merged_data = pd.merge(energy_consumption, solar_info, on=['Site Name', 'Day', 'Hour'])
#     merged_data = pd.merge(merged_data, site_info, on='Site Name')

#     # Display merged data
#     st.subheader('Merged Data Overview')
#     st.write(merged_data.head(20))
#     st.write(f"Shape: {merged_data.shape}")

#     # Convert Hour column to datetime datatype and extract hour
#     merged_data['Hour'] = pd.to_datetime(merged_data['Hour'], errors='coerce')
#     merged_data['Hour'] = merged_data['Hour'].dt.hour

#     # Features and Target selection
#     X = merged_data[['Solar Zenith Angle', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Relative Humidity', 'Day', 'Hour']]
#     y = merged_data['Energy Output(kWh)']

#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Train the model
#     solar_model = RandomForestRegressor(n_estimators=100, random_state=42)
#     solar_model.fit(X_train, y_train)

#     # Predict and display predictions
#     merged_data['predicted_solar_output'] = solar_model.predict(X)
#     st.subheader('Predicted Solar Output')
#     st.write(merged_data[['predicted_solar_output']].head())

#     # Optimization using PuLP
#     st.subheader('Optimization Problem Setup')

#     # Constants for the optimization problem
#     time_intervals = len(merged_data)
#     battery_capacity_kwh = 100
#     battery_threshold = 20
#     initial_battery_level = 50
#     cost_grid = 0.15
#     cost_solar = 0.05
#     cost_diesel = 0.20
#     predicted_solar = merged_data['predicted_solar_output'].values
#     demand = merged_data['Total Energy(kWh)'].values
#     grid_outage = merged_data['grid outage plan'].values

#     # Define LP problem
#     prob = lp.LpProblem("Minimize_Energy_Cost", lp.LpMinimize)

#     # Decision variables
#     usage_grid = [lp.LpVariable(f"usage_grid_{t}", 0, demand[t]) for t in range(time_intervals)]
    
#     # Add objective function and constraints (to be continued based on your logic)

#     st.write("Optimization model setup completed. Further constraints and solution logic can be implemented based on your project needs.")

# else:
#     st.warning('Please upload all required CSV files.')

