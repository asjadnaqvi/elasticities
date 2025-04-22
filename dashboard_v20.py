# Dashboard script for the paper "Global demand and supply elasticities and the impact of tariff shocks"
# https://www.wifo.ac.at/en/publication/424385/
# Asjad Naqvi (asjadnaqvi@gmail.com)
# Last updated 22 April 2025

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Set dashboard layout to wide
st.set_page_config(layout="wide")

# Tabs for navigation
main_tabs = st.tabs(["Dashboard", "Notes"])

with main_tabs[1]:
    st.markdown("This dashboard shows _provisional results_ from the **Global demand and supply elasticities and the impact of tariff shocks** (v1) working paper posted on 14th April 2025.")
    st.markdown("The paper can be downloaded either from the [WIFO Working Paper series](https://www.wifo.ac.at/en/publication/424385/) or [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5217187).")
    st.markdown("The paper estimates a [Quadratic Almost Ideal Demand System (QUAIDS)](https://en.wikipedia.org/wiki/Almost_ideal_demand_system) demand and supply elasticities using the 2021-2023 pooled data from the Asian Development Bank's (ADB) [Multi-Regional Input-Output (MRIO)](https://www.adb.org/what-we-do/data/regional-input-output-tables) database. The database covers 62 countries plus the Rest of the World and is available from 2007-2023.")

    st.markdown("The figure below summarizes the estimation strategy:")

    st.image("mrio_setup.png", caption="Demand estimation from the MRIO framework")

    st.markdown("The baseline model distinguishes between **Intermediate** versus **Final** demand goods supplied by **Domestic** or **Foreign** sectors representing a **2x2** system.")
    st.markdown("Additional results are presented in the paper including time series estimates and a detailed 6x2 sector decomposition. These will be added here in the future.")
    st.markdown("Please note that the paper also estimates the impact of tariff shocks using tariff data from 7th April 2025. Since this information is changing rapidly, results will be updated in the next version once tariff rates stabilize (currently planned for release in Summer 2025).")
    st.markdown("This project is supported by the [Supply Chain Intelligence Institute Austria (ASCII)](https://ascii.ac.at/). For comments and feedback either open an [Issue on GitHub](https://github.com/asjadnaqvi/elasticities/issues), or e-mail at asjad.naqvi@wifo.ac.at.")
    st.markdown("*This section was last updated on: 22 April 2025.*")    

with main_tabs[0]:

# Load data
# Variable labels for clarity in plots
    variable_labels = {
        "share1": "Intermediate - Domestic",
        "share2": "Intermediate - Foreign",
        "share3": "Final - Domestic",
        "share4": "Final - Foreign",
        "price1": "Intermediate - Domestic",
        "price2": "Intermediate - Foreign",
        "price3": "Final - Domestic",
        "price4": "Final - Foreign",
        "val1": "Intermediate - Domestic",
        "val2": "Intermediate - Foreign",
        "val3": "Final - Domestic",
        "val4": "Final - Foreign",
        "eta_1": "Intermediate - Domestic",
        "eta_2": "Intermediate - Foreign",
        "eta_3": "Final - Domestic",
        "eta_4": "Final - Foreign",
        
    }
    @st.cache_data
    def load_data():
        return pd.read_csv("elasticities_4sec_shock.csv")

    df = load_data()

    # Strip leading/trailing spaces and normalise case
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.strip().str.lower()

    # Strip spaces from country names to ensure exact matching in the dropdown
    if "name2" in df.columns:
        df["name2"] = df["name2"].astype(str).str.strip()


    # Split data into demand and supply
    if 'type' in df.columns:
        df_demand = df[df['type'] == 'demand']
        df_supply = df[df['type'] == 'supply']
    else:
        df_demand = df.copy()
        df_supply = df.copy()

    # Top dropdown control
    country = st.sidebar.selectbox("Select a Country", sorted(df["name2"].unique()))

    # Filter data for selected country
    country_supply = df_supply[df_supply["name2"] == country]
    supply_group = country_supply.copy()

    # Safety check for empty country selection
    df_countries = df["name2"].unique()
    if country not in df_countries:
        st.warning("Selected country has no data.")
        st.stop()
    country_data = df[df["name2"] == country].iloc[0:1]
    demand_group = df_demand[df_demand["name2"] == country]


    if demand_group.empty:
        st.warning("No demand‑side data available for the selected country.")
        # st.write(f"Rows in df_demand for '{country}':", len(df_demand[df_demand[\"name2\"] == country]))
        st.stop()

    # Extract values
    goods = ["Good 1", "Good 2", "Good 3", "Good 4"]

    # Percentiles (conditioned on selected country)
    expenditures_df = demand_group[["share1", "share2", "share3", "share4"]]
    p25_expenditures = [round(x, 3) for x in expenditures_df.quantile(0.25).tolist()]
    p75_expenditures = [round(x, 3) for x in expenditures_df.quantile(0.75).tolist()]

    # Prices for the selected country
    prices_df = demand_group[["price1", "price2", "price3", "price4"]]
    p25_prices = [round(x, 1) for x in prices_df.quantile(0.25).tolist()]
    p75_prices = [round(x, 1) for x in prices_df.quantile(0.75).tolist()]

    # Country-level mean values (weighted by total expenditure)
    total_val_demand = demand_group[["val1", "val2", "val3", "val4"]].sum().sum()
    total_val_supply = supply_group[["val1", "val2", "val3", "val4"]].sum().sum()

    mean_expenditures_demand = [
        round(demand_group["val1"].sum() / total_val_demand, 3) if total_val_demand != 0 else np.nan,
        round(demand_group["val2"].sum() / total_val_demand, 3) if total_val_demand != 0 else np.nan,
        round(demand_group["val3"].sum() / total_val_demand, 3) if total_val_demand != 0 else np.nan,
        round(demand_group["val4"].sum() / total_val_demand, 3) if total_val_demand != 0 else np.nan
    ]

    mean_expenditures_supply = [
        round(supply_group["val1"].sum() / total_val_supply, 3) if total_val_supply != 0 else np.nan,
        round(supply_group["val2"].sum() / total_val_supply, 3) if total_val_supply != 0 else np.nan,
        round(supply_group["val3"].sum() / total_val_supply, 3) if total_val_supply != 0 else np.nan,
        round(supply_group["val4"].sum() / total_val_supply, 3) if total_val_supply != 0 else np.nan
    ]

    mean_prices_demand = [
        round((demand_group["val1"] * demand_group["price1"]).sum() / demand_group["val1"].sum(), 2) if demand_group["val1"].sum() != 0 else np.nan,
        round((demand_group["val2"] * demand_group["price2"]).sum() / demand_group["val2"].sum(), 2) if demand_group["val2"].sum() != 0 else np.nan,
        round((demand_group["val3"] * demand_group["price3"]).sum() / demand_group["val3"].sum(), 2) if demand_group["val3"].sum() != 0 else np.nan,
        round((demand_group["val4"] * demand_group["price4"]).sum() / demand_group["val4"].sum(), 2) if demand_group["val4"].sum() != 0 else np.nan
    ] 

    mean_prices_supply= [
        round((supply_group["val1"] * supply_group["price1"]).sum() / supply_group["val1"].sum(), 2) if supply_group["val1"].sum() != 0 else np.nan,
        round((supply_group["val2"] * supply_group["price2"]).sum() / supply_group["val2"].sum(), 2) if supply_group["val2"].sum() != 0 else np.nan,
        round((supply_group["val3"] * supply_group["price3"]).sum() / supply_group["val3"].sum(), 2) if supply_group["val3"].sum() != 0 else np.nan,
        round((supply_group["val4"] * supply_group["price4"]).sum() / supply_group["val4"].sum(), 2) if supply_group["val4"].sum() != 0 else np.nan
    ] 

    # Global mean values (weighted by total expenditure)
    total_val_global = df[["val1", "val2", "val3", "val4"]].sum().sum()

    total_val_global_demand = df_demand[["val1", "val2", "val3", "val4"]].sum().sum()
    total_val_global_supply = df_supply[["val1", "val2", "val3", "val4"]].sum().sum()

    global_mean_expenditures_demand = [
        round(df_demand["val1"].sum() / total_val_global_demand, 3) if total_val_global_demand != 0 else np.nan,
        round(df_demand["val2"].sum() / total_val_global_demand, 3) if total_val_global_demand != 0 else np.nan,
        round(df_demand["val3"].sum() / total_val_global_demand, 3) if total_val_global_demand != 0 else np.nan,
        round(df_demand["val4"].sum() / total_val_global_demand, 3) if total_val_global_demand != 0 else np.nan
    ]

    global_mean_expenditures_supply = [
        round(df_supply["val1"].sum() / total_val_global_supply, 3) if total_val_global_supply != 0 else np.nan,
        round(df_supply["val2"].sum() / total_val_global_supply, 3) if total_val_global_supply != 0 else np.nan,
        round(df_supply["val3"].sum() / total_val_global_supply, 3) if total_val_global_supply != 0 else np.nan,
        round(df_supply["val4"].sum() / total_val_global_supply, 3) if total_val_global_supply != 0 else np.nan
    ]


    global_mean_prices_demand = [
        round((df_demand["val1"] * df_demand["price1"]).sum() / df_demand["val1"].sum(), 2) if df_demand["val1"].sum() != 0 else np.nan,
        round((df_demand["val2"] * df_demand["price2"]).sum() / df_demand["val2"].sum(), 2) if df_demand["val2"].sum() != 0 else np.nan,
        round((df_demand["val3"] * df_demand["price3"]).sum() / df_demand["val3"].sum(), 2) if df_demand["val3"].sum() != 0 else np.nan,
        round((df_demand["val4"] * df_demand["price4"]).sum() / df_demand["val4"].sum(), 2) if df_demand["val4"].sum() != 0 else np.nan
    ] 

    global_mean_prices_supply = [
        round((df_supply["val1"] * df_supply["price1"]).sum() / df_supply["val1"].sum(), 2) if df_supply["val1"].sum() != 0 else np.nan,
        round((df_supply["val2"] * df_supply["price2"]).sum() / df_supply["val2"].sum(), 2) if df_supply["val2"].sum() != 0 else np.nan,
        round((df_supply["val3"] * df_supply["price3"]).sum() / df_supply["val3"].sum(), 2) if df_supply["val3"].sum() != 0 else np.nan,
        round((df_supply["val4"] * df_supply["price4"]).sum() / df_supply["val4"].sum(), 2) if df_supply["val4"].sum() != 0 else np.nan
    ] 

    # Manually define the scale values

    # Expenditure Shares
    global_min_share = 0.0
    global_max_share = 0.6

    # Prices
    global_min_price = 0.5
    global_max_price = 2

    # Price Elasticities
    global_min_epsilon = -1.5
    global_max_epsilon = 1.5
    
    # Expenditure Elasticities
    global_min_eta =  0.5
    global_max_eta =  1.5

    # Toggle view mode
    view_mode = st.sidebar.radio("Select View Mode:", ["Heatmaps", "Detailed Plots"], horizontal=False)

    epsilon_cols = [f"epsilon_{p}_{q}" for q in range(1, 5) for p in range(1, 5)]
    valid_epsilons = [col for col in epsilon_cols if col in demand_group.columns or col in supply_group.columns]

    combined_epsilon_data = pd.concat(
        [
            demand_group[[col for col in valid_epsilons if col in demand_group.columns]],
            supply_group[[col for col in valid_epsilons if col in supply_group.columns]]
        ],
        axis=0
    )

    st.markdown("### Expenditure Shares and Prices")
    st.markdown('The plots below show expenditure shares and prices (unit costs) split by demand and supply extracted from the pooled 2021-2023 MRIO data (see Notes tab above). Heatmaps shows averages while Detailed plots show the full data distribution.')
    st.markdown("Average shares add up to one in the demand and supply columns respectively. Prices are calculated as nominal over real values and are effectively relative unit values.")

    if view_mode == "Detailed Plots":
        
        # Top row: Expenditure Shares (Demand and Supply)
        col1, col2 = st.columns([1, 1])

        with col1:
            # Expenditure Shares (Demand)
            fig1 = go.Figure()
            for i, good in enumerate(goods):
                fig1.add_trace(go.Box(
                    y=demand_group[f"share{i+1}"].dropna(),
                    name=variable_labels.get(f"share{i+1}", good),
                    boxpoints='outliers',
                    marker_color='goldenrod',
                    showlegend=False
                ))
            for i, good in enumerate(goods):
                fig1.add_trace(go.Scatter(
                    x=[variable_labels.get(f"share{i+1}", good)],
                    y=[mean_expenditures_demand[i]],
                    mode='markers',
                    name='Country Mean' if i == 0 else None,
                    showlegend=(i == 0),
                    marker=dict(color='darkblue', symbol='circle', size=10)
                ))
            for i, good in enumerate(goods):
                fig1.add_trace(go.Scatter(
                    x=[variable_labels.get(f"share{i+1}", good)],
                    y=[global_mean_expenditures_demand[i]],
                    mode='markers',
                    name='Global Mean' if i == 0 else None,
                    showlegend=(i == 0),
                    marker=dict(color='darkred', symbol='x', size=10)
                ))
            fig1.update_layout(
                title="Expenditure Shares (Demand)",
                yaxis_title="Share",
                yaxis_tickformat=".2f",
                width=1000,
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig1, use_container_width=True, key="fig1")

        with col2:
            # Expenditure Shares (Supply)
            fig3 = go.Figure()
            for i, good in enumerate(goods):
                fig3.add_trace(go.Box(
                    y=supply_group[f"share{i+1}"].dropna(),
                    name=variable_labels.get(f"share{i+1}", good),
                    boxpoints='outliers',
                    marker_color='goldenrod',
                    showlegend=False
                ))
            for i, good in enumerate(goods):
                fig3.add_trace(go.Scatter(
                    x=[variable_labels.get(f"share{i+1}", good)],
                    y=[mean_expenditures_supply[i]],
                    mode='markers',
                    name='Country Mean' if i == 0 else None,
                    showlegend=(i == 0),
                    marker=dict(color='darkblue', symbol='circle', size=10)
                ))
            for i, good in enumerate(goods):
                fig3.add_trace(go.Scatter(
                    x=[variable_labels.get(f"share{i+1}", good)],
                    y=[global_mean_expenditures_supply[i]],
                    mode='markers',
                    name='Global Mean' if i == 0 else None,
                    showlegend=(i == 0),
                    marker=dict(color='darkred', symbol='x', size=10)
                ))
            fig3.update_layout(
                title="Expenditure Shares (Supply)",
                # shapes=[dict(type="line", y0=0.5, y1=0.5, x0=-0.5, x1=3.5, line=dict(color="black", dash="solid", width=0.4))],
                yaxis_title="Share",
                yaxis_tickformat=".2f",
                width=1000,
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig3, use_container_width=True, key="fig3_exp_shares_supply")

        # Second row: Prices (Demand and Supply)
        col3, col4 = st.columns([1, 1])

        with col3:
            fig2 = go.Figure()
            for i, good in enumerate(goods):
                fig2.add_trace(go.Box(
                    y=demand_group[f"price{i+1}"].dropna(),
                    name=variable_labels.get(f"price{i+1}", good),
                    boxpoints='outliers',
                    marker_color='goldenrod',
                    showlegend=False
                ))
            for i, good in enumerate(goods):
                fig2.add_trace(go.Scatter(
                    x=[variable_labels.get(f"price{i+1}", good)],
                    y=[mean_prices_demand[i]],
                    mode='markers',
                    name='Country Mean' if i == 0 else None,
                    showlegend=(i == 0),
                    marker=dict(color='darkblue', symbol='circle', size=10)
                ))
            for i, good in enumerate(goods):
                fig2.add_trace(go.Scatter(
                    x=[variable_labels.get(f"price{i+1}", good)],
                    y=[global_mean_prices_demand[i]],
                    mode='markers',
                    name='Global Mean' if i == 0 else None,
                    showlegend=(i == 0),
                    marker=dict(color='darkred', symbol='x', size=10)
                ))
            fig2.update_layout(
                title="Prices (Demand)",
                yaxis_range=(
                    min(pd.concat([demand_group[["price1", "price2", "price3", "price4"]], supply_group[["price1", "price2", "price3", "price4"]]]).min()) - 0.2,
                    max(pd.concat([demand_group[["price1", "price2", "price3", "price4"]], supply_group[["price1", "price2", "price3", "price4"]]]).max()) + 0.2
                ),
                yaxis_title="Price",
                yaxis_tickformat=".2f",
                width=1000,
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig2, use_container_width=True, key="fig2_prices_demand")

        with col4:
            fig4 = go.Figure()
            for i, good in enumerate(goods):
                fig4.add_trace(go.Box(
                    y=supply_group[f"price{i+1}"].dropna(),
                    name=variable_labels.get(f"price{i+1}", good),
                    boxpoints='outliers',
                    marker_color='goldenrod',
                    showlegend=False
                ))
            for i, good in enumerate(goods):
                fig4.add_trace(go.Scatter(
                    x=[variable_labels.get(f"price{i+1}", good)],
                    y=[mean_prices_supply[i]],
                    mode='markers',
                    name='Country Mean' if i == 0 else None,
                    showlegend=(i == 0),
                    marker=dict(color='darkblue', symbol='circle', size=10)
                ))
            for i, good in enumerate(goods):
                fig4.add_trace(go.Scatter(
                    x=[variable_labels.get(f"price{i+1}", good)],
                    y=[global_mean_prices_supply[i]],
                    mode='markers',
                    name='Global Mean' if i == 0 else None,
                    showlegend=(i == 0),
                    marker=dict(color='darkred', symbol='x', size=10)
                ))
            fig4.update_layout(
                title="Prices (Supply)",
                yaxis_range=(
                    min(pd.concat([demand_group[["price1", "price2", "price3", "price4"]], supply_group[["price1", "price2", "price3", "price4"]]]).min()) - 0.2,
                    max(pd.concat([demand_group[["price1", "price2", "price3", "price4"]], supply_group[["price1", "price2", "price3", "price4"]]]).max()) + 0.2
                ),
                yaxis_title="Price",
                yaxis_tickformat=".2f",
                width=1000,
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig4, use_container_width=True, key="fig4")

    # Third row: Expenditure Elasticities

    elasticities_df = demand_group[["eta_1", "eta_2", "eta_3", "eta_4"]]
    mean_elasticities = []
    for i in range(4):
        col = f"eta_{i+1}"
        if col in demand_group.columns and not demand_group[col].dropna().empty:
            mean_elasticities.append(round(demand_group[col].mean(), 2))
        else:
            mean_elasticities.append(np.nan)

    global_mean_elasticities = []
    for i in range(4):
        col = f"eta_{i+1}"
        if col in df.columns and not df[col].dropna().empty:
            global_mean_elasticities.append(round(df[col].mean(), 2))
        else:
            global_mean_elasticities.append(np.nan)


    eta1 = go.Figure()
    for i, good in enumerate(goods):
        eta_col = f"eta_{i+1}"
        if eta_col in demand_group.columns:
            filtered_eta = demand_group[eta_col][(demand_group[eta_col] >= -3) & (demand_group[eta_col] <= 3)].dropna()
            eta1.add_trace(go.Box(
                y=filtered_eta,
                name=variable_labels.get(f"eta_{i+1}", good),
                boxpoints='outliers',
                marker_color='goldenrod',
                showlegend=False
            ))
    for i, good in enumerate(goods):
        if not np.isnan(mean_elasticities[i]):
            eta1.add_trace(go.Scatter(
                x=[variable_labels.get(f"eta_{i+1}", good)],
                    y=[mean_elasticities[i]],
                mode='markers',
                name='Country Mean' if i == 0 else None,
                showlegend=(i == 0),
                marker=dict(color='darkblue', symbol='circle', size=10)
            ))
        if not np.isnan(global_mean_elasticities[i]):
            eta1.add_trace(go.Scatter(
                x=[variable_labels.get(f"eta_{i+1}", good)],
                y=[global_mean_elasticities[i]],
                mode='markers',
                name='Global Mean' if i == 0 else None,
                showlegend=(i == 0),
                marker=dict(color='darkred', symbol='x', size=10)
            ))
    eta1.update_layout(
        yaxis_range=(
            min(pd.concat([demand_group[["eta_1", "eta_2", "eta_3", "eta_4"]], supply_group[["eta_1", "eta_2", "eta_3", "eta_4"]]]).min()) - 0.2,
            max(pd.concat([demand_group[["eta_1", "eta_2", "eta_3", "eta_4"]], supply_group[["eta_1", "eta_2", "eta_3", "eta_4"]]]).max()) + 0.2
        ),
        title="Expenditure Elasticities (Demand)",
        yaxis_title="Elasticity",
        shapes=[dict(type="line", y0=1, y1=1, x0=-0.5, x1=3.5, line=dict(color="black", dash="solid", width=0.4))],
        yaxis_tickformat=".2f",
        width=1000,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )


    # Expenditure elasticities - supply

    elasticities_df2 = supply_group[["eta_1", "eta_2", "eta_3", "eta_4"]]
    mean_elasticities2 = []

    for i in range(4):
        col = f"eta_{i+1}"
        if col in supply_group.columns and not supply_group[col].dropna().empty:
            mean_elasticities2.append(round(supply_group[col].mean(), 2))
        else:
            mean_elasticities2.append(np.nan)

    global_mean_elasticities2 = []
    for i in range(4):
        col = f"eta_{i+1}"
        if col in df.columns and not df[col].dropna().empty:
            global_mean_elasticities2.append(round(df[col].mean(), 2))
        else:
            global_mean_elasticities2.append(np.nan)

    eta2 = go.Figure()
    for i, good in enumerate(goods):
        eta_col = f"eta_{i+1}"
        if eta_col in supply_group.columns:
            filtered_eta2 = supply_group[eta_col][(supply_group[eta_col] >= -3) & (supply_group[eta_col] <= 3)].dropna()
            eta2.add_trace(go.Box(
                y=filtered_eta2,
                name=variable_labels.get(f"eta_{i+1}", good),
                boxpoints='outliers',
                marker_color='goldenrod',
                showlegend=False
            ))
    for i, good in enumerate(goods):
        if not np.isnan(mean_elasticities2[i]):
            eta2.add_trace(go.Scatter(
                x=[variable_labels.get(f"eta_{i+1}", good)],
                y=[mean_elasticities2[i]],
                mode='markers',
                name='Country Mean' if i == 0 else None,
                showlegend=(i == 0),
                marker=dict(color='darkblue', symbol='circle', size=10)
            ))
        if not np.isnan(global_mean_elasticities2[i]):
            eta2.add_trace(go.Scatter(
                x=[variable_labels.get(f"eta_{i+1}", good)],
                y=[global_mean_elasticities2[i]],
                mode='markers',
                name='Global Mean' if i == 0 else None,
                showlegend=(i == 0),
                marker=dict(color='darkred', symbol='x', size=10)
            ))
    eta2.update_layout(
        yaxis_range=(
            min(pd.concat([demand_group[["eta_1", "eta_2", "eta_3", "eta_4"]], supply_group[["eta_1", "eta_2", "eta_3", "eta_4"]]]).min()) - 0.2,
            max(pd.concat([demand_group[["eta_1", "eta_2", "eta_3", "eta_4"]], supply_group[["eta_1", "eta_2", "eta_3", "eta_4"]]]).max()) + 0.2
        ),
        title="Expenditure Elasticities (Supply)",
        shapes=[dict(type="line", y0=1, y1=1, x0=-0.5, x1=3.5, line=dict(color="black", dash="solid", width=0.4))],
        yaxis_title="Elasticity",
        yaxis_tickformat=".2f",
        width=1000,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )

    if view_mode == "Heatmaps":
        # st.markdown("### Heatmap Summaries")

        top_row_1 = st.container()
        #bot_row_1, bot_row_2 = st.columns([1, 1])

        with top_row_1:
            # st.markdown("### Mean Shares and Prices")
            share_col, price_col = st.columns(2)

        with share_col:
            # supply_group already defined globally; duplicate removed
            if all(col in supply_group.columns for col in ["val1", "val2", "val3", "val4"]):
                total_val_supply = supply_group[["val1", "val2", "val3", "val4"]].sum().sum()
                mean_expenditures_supply = [
                    round(supply_group["val1"].sum() / total_val_supply, 3) if total_val_supply != 0 else np.nan,
                    round(supply_group["val2"].sum() / total_val_supply, 3),
                    round(supply_group["val3"].sum() / total_val_supply, 3),
                    round(supply_group["val4"].sum() / total_val_supply, 3)
                ]
            elif all(f"share{i+1}" in supply_group.columns for i in range(4)):
                mean_expenditures_supply = [
                    round(supply_group[f"share{i+1}"].mean(), 3) for i in range(4)
                ]
            else:
                st.warning("Supply data for expenditure shares is not available for this country.")
                mean_expenditures_supply = [np.nan] * 4

            combined_data = np.column_stack([mean_expenditures_demand, mean_expenditures_supply])
            fig_share_combined = px.imshow(
                combined_data,
                zmin=global_min_share,
                zmax=global_max_share,
                color_continuous_scale='YlGnBu',
                x=["Demand", "Supply"],
                y=[variable_labels.get(f"share{i+1}", good) for i, good in enumerate(goods)],
                text_auto=".2f",
                labels=dict(color="Share")
            )
            fig_share_combined.update_layout(height=400, title="Expenditure Shares")
            st.plotly_chart(fig_share_combined, use_container_width=True, key="fig_share_combined")

        with price_col:
            if all(col in supply_group.columns for col in ["val1", "val2", "val3", "val4"]):
                total_val_supply = supply_group[["val1", "val2", "val3", "val4"]].sum().sum()
                mean_prices_supply = [
                    round((supply_group["val1"] * supply_group["price1"]).sum() / supply_group["val1"].sum(), 2) if supply_group["val1"].sum() != 0 else np.nan,
                    round((supply_group["val2"] * supply_group["price2"]).sum() / supply_group["val2"].sum(), 2),
                    round((supply_group["val3"] * supply_group["price3"]).sum() / supply_group["val3"].sum(), 2),
                    round((supply_group["val4"] * supply_group["price4"]).sum() / supply_group["val4"].sum(), 2)
                ]
            else:
                mean_prices_supply = [np.nan] * 4

            combined_price_data = np.column_stack([mean_prices_demand, mean_prices_supply])
            fig_price_combined = px.imshow(
                combined_price_data,
                zmin=global_min_price,
                zmax=global_max_price,
                x=["Demand", "Supply"],
                y=[variable_labels.get(f"price{i+1}", good) for i, good in enumerate(goods)],
                color_continuous_scale='YlGnBu',
                text_auto=".2f",
                labels=dict(color="Price")
            )
            fig_price_combined.update_layout(height=400, title="Prices")
            st.plotly_chart(fig_price_combined, use_container_width=True, key="fig_price_combined")


    st.markdown("### Elasticities")
    st.markdown("Elasticities are generated from a QUAIDS model using the pooled 2021-2023 MRIO data (see Notes). Results are split into Expenditure ($\\eta$) and Uncompensated Price ($\\epsilon^c$) elasticities. These are further split by demand and supply groups. Heatmaps show averages while Detailed Plots show the full distribution of the estimates.")

    st.markdown("Expenditure elasticities: $\\eta >1$ is a luxury good, $\\eta \\in [0,1]$ is a normal good, $\\eta<0$ is an inferior good.")
    st.markdown("Own-price elasticities (diagonal): $\\epsilon_{ii} < -1$ is elastic demand, $\\epsilon_{ii} \\in [-1,0]$ is inelastic demand.")
    st.markdown("Cross-price elasticities (off-diagonal): $\\epsilon_{ij} < 0$ are substitutes, $\\epsilon_{ij} > 0$ are complements.")

    
    if view_mode == "Detailed Plots":

        eta_col1, eta_col2 = st.columns([1, 1])

        with eta_col1:
            st.plotly_chart(eta1, use_container_width=True, key="eta_demand")
        with eta_col2:
            st.plotly_chart(eta2, use_container_width=True, key="eta_supply")

        # st.markdown("### Price Elasticities")
        col_price_demand_box, col_price_supply_box = st.columns(2)

        with col_price_demand_box:
            for q in range(1, 5):
                fig_demand = go.Figure()
                for p in range(1, 5):
                    epsilon_col = f"epsilon_{p}_{q}"
                    if epsilon_col in demand_group.columns:
                        filtered_data = demand_group[epsilon_col][(demand_group[epsilon_col] >= -3) & (demand_group[epsilon_col] <= 3)].dropna()
                        fig_demand.add_trace(go.Box(
                            y=filtered_data,
                            name=variable_labels.get(f"price{p}", f"Price {p}"),
                            boxpoints='outliers',
                            marker_color='goldenrod',
                            showlegend=False
                        ))
                        country_mean = demand_group[epsilon_col].mean()
                        global_mean = df[epsilon_col].mean()
                        fig_demand.add_trace(go.Scatter(
                            x=[variable_labels.get(f"price{p}", f"Price {p}")],
                            y=[country_mean],
                            mode='markers',
                            name='Country Mean' if p == 1 else None,
                            showlegend=(p == 1),
                            marker=dict(color='darkblue', symbol='circle', size=10)
                        ))
                        fig_demand.add_trace(go.Scatter(
                            x=[variable_labels.get(f"price{p}", f"Price {p}")],
                            y=[global_mean],
                            mode='markers',
                            name='Global Mean' if p == 1 else None,
                            showlegend=(p == 1),
                            marker=dict(color='darkred', symbol='x', size=10)
                        ))
                fig_demand.update_layout(
                    shapes=[dict(type="line", y0=0, y1=0, x0=-0.5, x1=3.5, line=dict(color="black", dash="solid", width=0.4))],
                    yaxis_range=(
                        min(combined_epsilon_data.min()) - 0.2,
                        max(combined_epsilon_data.max()) + 0.2
                    ),
                    yaxis_title=f"Elasticity",
                    width=500,
                    height=400,
                    yaxis_tickformat=".2f",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                    title=f"{variable_labels.get(f'val{q}', f'Good {q}')} (Demand)"
                )
                st.plotly_chart(fig_demand, use_container_width=True, key=f"fig_price_demand_{q}")

        with col_price_supply_box:
            for q in range(1, 5):
                fig_supply = go.Figure()
                for p in range(1, 5):
                    epsilon_col = f"epsilon_{p}_{q}"
                    if epsilon_col in supply_group.columns:
                        filtered_data = supply_group[epsilon_col][(supply_group[epsilon_col] >= -3) & (supply_group[epsilon_col] <= 3)].dropna()
                        fig_supply.add_trace(go.Box(
                            y=filtered_data,
                            name=variable_labels.get(f"price{p}", f"Price {p}"),
                            boxpoints='outliers',
                            marker_color='goldenrod',
                            showlegend=False
                        ))
                        country_mean = supply_group[epsilon_col].mean()
                        global_mean = df[epsilon_col].mean()
                        fig_supply.add_trace(go.Scatter(
                            x=[variable_labels.get(f"price{p}", f"Price {p}")],
                            y=[country_mean],
                            mode='markers',
                            name='Country Mean' if p == 1 else None,
                            showlegend=(p == 1),
                            marker=dict(color='darkblue', symbol='circle', size=10)
                        ))
                        fig_supply.add_trace(go.Scatter(
                            x=[variable_labels.get(f"price{p}", f"Price {p}")],
                            y=[global_mean],
                            mode='markers',
                            name='Global Mean' if p == 1 else None,
                            showlegend=(p == 1),
                            marker=dict(color='darkred', symbol='x', size=10)
                        ))
                fig_supply.update_layout(
                    shapes=[dict(type="line", y0=0, y1=0, x0=-0.5, x1=3.5, line=dict(color="black", dash="solid", width=0.4))],
                    yaxis_range=(
                        min(combined_epsilon_data.min()) - 0.2,
                        max(combined_epsilon_data.max()) + 0.2
                    ),
                    yaxis_title=f"Elasticity",
                    width=500,
                    height=400,
                    yaxis_tickformat=".2f",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                    title=f"{variable_labels.get(f'val{q}', f'Good {q}')} (Supply)"
                )
                st.plotly_chart(fig_supply, use_container_width=True, key=f"fig_price_supply_{q}")



    if view_mode == "Heatmaps":

        bot_row_1 = st.container()
        # bot_row_1 = st.columns([1, 1])

        with bot_row_1:
            # st.markdown("### Mean Expenditure Elasticity")
            supply_eta_array = np.array(mean_elasticities2).reshape(-1, 1)
            eta_means = np.column_stack([mean_elasticities, mean_elasticities2])
            fig_eta_heat = px.imshow(
                eta_means,
                x=["Demand", "Supply"],
                y=[variable_labels.get(f"share{i+1}", good) for i, good in enumerate(goods)],
                color_continuous_scale='RdBu',
                zmin=global_min_eta,
                zmax=global_max_eta,
                text_auto=".2f",
                labels=dict(color="Elasticity")
            )
            fig_eta_heat.update_layout(height=400, title="Expenditure Elasticities")
            st.plotly_chart(fig_eta_heat)


        price_elasticity_row = st.container()
        with price_elasticity_row:
            col_price_demand, col_price_supply = st.columns(2)

            with col_price_demand:
                heatmap_data = np.zeros((4, 4))
                for i in range(4):
                    for j in range(4):
                        col = f"epsilon_{j+1}_{i+1}"
                        if col in demand_group.columns:
                            filtered_vals = demand_group[col][(demand_group[col] >= -3) & (demand_group[col] <= 3)].dropna()
                            heatmap_data[i, j] = filtered_vals.mean()
                        else:
                            heatmap_data[i, j] = np.nan
                fig_heat = px.imshow(
                    heatmap_data,
                    x=[variable_labels.get(f"price{i+1}", f"P{i+1}") for i in range(4)],
                    y=[variable_labels.get(f"val{i+1}", f"Q{i+1}") for i in range(4)],
                    color_continuous_scale='RdBu',
                    zmin=global_min_epsilon, zmax=global_max_epsilon,
                    aspect="auto",
                    text_auto=".2f",
                    labels=dict(x="Price", y="Quantity", color="Elasticity")
                )
                fig_heat.update_layout(height=400, title="Uncompensated Price Elasticities (Demand)")
                st.plotly_chart(fig_heat, use_container_width=True, key="fig_price_demand")

            with col_price_supply:
                heatmap_data_supply = np.zeros((4, 4))
                for i in range(4):
                    for j in range(4):
                        col = f"epsilon_{j+1}_{i+1}"
                        if col in supply_group.columns:
                            filtered_vals = supply_group[col][(supply_group[col] >= -3) & (supply_group[col] <= 3)].dropna()
                            heatmap_data_supply[i, j] = filtered_vals.mean()
                        else:
                            heatmap_data_supply[i, j] = np.nan
                fig_heat_supply = px.imshow(
                    heatmap_data_supply,
                    x=[variable_labels.get(f"price{i+1}", f"P{i+1}") for i in range(4)],
                    y=[variable_labels.get(f"val{i+1}", f"Q{i+1}") for i in range(4)],
                    color_continuous_scale='RdBu',
                    zmin=global_min_epsilon, zmax=global_max_epsilon,
                    aspect="auto",
                    text_auto=".2f",
                    labels=dict(x="Price", y="Quantity", color="Elasticity")
                )
                fig_heat_supply.update_layout(height=400, title="Uncompensated Price Elasticities (Supply)")
                st.plotly_chart(fig_heat_supply, use_container_width=True, key="fig_price_supply")
