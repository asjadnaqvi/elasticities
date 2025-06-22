# Dashboard script for the paper "Global demand and supply elasticities and the impact of tariff shocks"
# https://www.wifo.ac.at/en/publication/424385/
# Asjad Naqvi (asjadnaqvi@gmail.com)
# Last updated 22 April 2025

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Set dashboard layout to wide and sidebar expanded
st.set_page_config(layout="wide", page_title="Global Demand and Supply Elasticities Dashboard", initial_sidebar_state="expanded")

# Add dashboard title
st.title("Global Demand and Supply Elasticities")

st.markdown("This dashboard shows provisional results from the [Global demand and supply elasticities and the impact of tariff shocks](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5217187) (*v1, 14 Apr 2025*) working paper. The dashboard was last updated on *22 Jun 2025*. For more information and feedback, please contact Asjad Naqvi (asjad.naqvi@wifo.ac.at).")

# Tabs for navigation
main_tabs = st.tabs(["Dashboard", "About"])


# About tab
#with main_tabs[1]:
#    st.markdown("This dashboard shows _provisional results_ from the **Global demand and supply elasticities and the impact of tariff shocks** (v1) working paper posted on 14th April 2025.")
#    st.markdown("The paper can be downloaded either from the [WIFO Working Paper series](https://www.wifo.ac.at/en/publication/424385/) or [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5217187).")
#    st.markdown("The paper estimates a [Quadratic Almost Ideal Demand System (QUAIDS)](https://en.wikipedia.org/wiki/Almost_ideal_demand_system) demand and supply elasticities using the 2021-2023 pooled data from the Asian Development Bank's (ADB) [Multi-Regional Input-Output (MRIO)](https://www.adb.org/what-we-do/data/regional-input-output-tables) database. The database covers 62 countries plus the Rest of the World and is available from 2007-2023.")

#    st.markdown("The figure below summarizes the estimation strategy:")

#    st.image("mrio_setup.png", caption="Demand estimation from the MRIO framework")

#    st.markdown("The baseline model distinguishes between **Intermediate** versus **Final** demand goods supplied by **Domestic** or **Foreign** sectors representing a **2x2** system.")
#    st.markdown("Additional results are presented in the paper including time series estimates and a detailed 6x2 sector decomposition. These will be added here in the future.")
#    st.markdown("Please note that the paper also estimates the impact of tariff shocks using tariff data from 7th April 2025. Since this information is changing rapidly, results will be updated in the next version once tariff rates stabilize (currently planned for release in Summer 2025).")
#    st.markdown("This project is supported by the [Supply Chain Intelligence Institute Austria (ASCII)](https://ascii.ac.at/). For comments and feedback either open an [Issue on GitHub](https://github.com/asjadnaqvi/elasticities/issues), or e-mail at asjad.naqvi@wifo.ac.at.")
#    st.markdown("*This section was last updated on: 22 April 2025.*")    


# FAQs tab
with main_tabs[1]:
    st.markdown("## Frequently Asked Questions (FAQs)")

    #st.markdown("For comments and feedback, you can either open an [Issue on GitHub](https://github.com/asjadnaqvi/elasticities/issues) or email me at asjad.naqvi@wifo.ac.at or asjadnaqvi@gmail.com.")

    with st.expander("What is the purpose of this dashboard?"):
        st.markdown("This dashboard provides a visual representation of the provisional results of the paper titled **Global Demand and Supply Elasticities and the Impact of Tariff Shocks**. The paper can be downloaded from [WIFO Working Paper series](https://www.wifo.ac.at/en/publication/424385/) or [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5217187).")

    with st.expander("What is the source of the data?"):
        st.markdown("The data is sourced from the Asian Development Bank's (ADB) [Multi-Regional Input-Output (MRIO)](https://www.adb.org/what-we-do/data/regional-input-output-tables) tables that are available from 2007-2023 and covers 62 countries plus the Rest of the World (RoW). To tbe best of my knowledge, this is currently the only MRIO database that provides stable real and nominal values that are required for elasticity estimation.")

    with st.expander("Which years are used for the analysis shown on the dashboard?"):
        st.markdown("The analysis uses pooled 2021-2023 data from the ADB's MRIO database. The reason to use pooled data is to ensure there are enough observations for estimating a fairly rigorous estimation procedure. Additionally, some sectors such as Energy and Services have few cross-country observations. Therefore, pooling helps minimizing spurious results that could arise from using a single year of data. Estimates for all three-year rolling periods (2007-2009, 2011-2013, 2025-2018, etc) are also available upon request. Results from different aggregations (1 year, 5 years, all years) are also discussed in the paper but not currently included in the dashboard. These might be added in a future version of the dashboard but are also available upon request.")

    with st.expander("How are elasticities calculated?"):
        st.markdown("Elasticities are estimated using the [Quadratic Almost Ideal Demand System (QUAIDS)](https://en.wikipedia.org/wiki/Almost_ideal_demand_system) model, which allows for flexible demand and supply elasticity estimation. Elasticity estimates from the *Translog*, the *Linear Expenditure System (LES)*, *Cobb Douglas*, and *AIDS* demand systems are also discussed in the paper but not currently included in the dashboard. These might be added in a future version of the dashboard but are available upon request.")

    with st.expander("What is the estimation strategy?"):
        st.markdown("The estimation strategy is based on the pooled 2021-2023 MRIO data and uses a **2x2** system to distinguish between **Intermediate** versus **Final** demand goods supplied by **Domestic** or **Foreign** sectors. The estimation is done using a QUAIDS model, which allows for flexible demand and supply elasticity estimation. Additional results for a detailed 6x2 sector decomposition are also discussed in the paper. These might be added to this dashboard in the future. The figure below summarizes the estimation strategy:")

        st.image("mrio_setup.png", caption="Demand estimation from the MRIO framework")

    with st.expander("How often are the results updated?"):
        st.markdown("The esults are expected to be updated at least once a year depending on how often the MRIO database is updated or whether other comprable datasets become available.")
    
    with st.expander("Who can I contact for feedback?"):
        st.markdown("For comments and feedback, you can either open an [Issue on GitHub](https://github.com/asjadnaqvi/elasticities/issues) or email me at asjad.naqvi@wifo.ac.at or asjadnaqvi@gmail.com.")


    st.markdown("This project is supported by the [Supply Chain Intelligence Institute Austria (ASCII)](https://ascii.ac.at/). For comments and feedback either open an [Issue on GitHub](https://github.com/asjadnaqvi/elasticities/issues), or e-mail at asjad.naqvi@wifo.ac.at or asjadnaqvi@gmail.com.")
    st.markdown("*Last updated on: 12 June 2025.*")  


# Main dashboard

with main_tabs[0]:

    @st.cache_data
    def load_data():
        return pd.read_csv("elasticities_4sec_shock.csv")

    df = load_data()

    # Strip leading/trailing spaces and normalise case
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.strip().str.lower()

    if "name2" in df.columns:
        df["name2"] = df["name2"].astype(str).str.strip()

    # Now place the selectbox here, after df and df['name2'] are ready
    country = st.selectbox("Select a Country", sorted(df["name2"].unique()))



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


    # Split data into demand and supply
    if 'type' in df.columns:
        df_demand = df[df['type'] == 'demand']
        df_supply = df[df['type'] == 'supply']
    else:
        df_demand = df.copy()
        df_supply = df.copy()

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
        round((demand_group["val1"] * demand_group["price1"]).sum() / demand_group["val1"].sum(), 3) if demand_group["val1"].sum() != 0 else np.nan,
        round((demand_group["val2"] * demand_group["price2"]).sum() / demand_group["val2"].sum(), 3) if demand_group["val2"].sum() != 0 else np.nan,
        round((demand_group["val3"] * demand_group["price3"]).sum() / demand_group["val3"].sum(), 3) if demand_group["val3"].sum() != 0 else np.nan,
        round((demand_group["val4"] * demand_group["price4"]).sum() / demand_group["val4"].sum(), 3) if demand_group["val4"].sum() != 0 else np.nan
    ] 

    mean_prices_supply= [
        round((supply_group["val1"] * supply_group["price1"]).sum() / supply_group["val1"].sum(), 3) if supply_group["val1"].sum() != 0 else np.nan,
        round((supply_group["val2"] * supply_group["price2"]).sum() / supply_group["val2"].sum(), 3) if supply_group["val2"].sum() != 0 else np.nan,
        round((supply_group["val3"] * supply_group["price3"]).sum() / supply_group["val3"].sum(), 3) if supply_group["val3"].sum() != 0 else np.nan,
        round((supply_group["val4"] * supply_group["price4"]).sum() / supply_group["val4"].sum(), 3) if supply_group["val4"].sum() != 0 else np.nan
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
        round((df_demand["val1"] * df_demand["price1"]).sum() / df_demand["val1"].sum(), 3) if df_demand["val1"].sum() != 0 else np.nan,
        round((df_demand["val2"] * df_demand["price2"]).sum() / df_demand["val2"].sum(), 3) if df_demand["val2"].sum() != 0 else np.nan,
        round((df_demand["val3"] * df_demand["price3"]).sum() / df_demand["val3"].sum(), 3) if df_demand["val3"].sum() != 0 else np.nan,
        round((df_demand["val4"] * df_demand["price4"]).sum() / df_demand["val4"].sum(), 3) if df_demand["val4"].sum() != 0 else np.nan
    ] 

    global_mean_prices_supply = [
        round((df_supply["val1"] * df_supply["price1"]).sum() / df_supply["val1"].sum(), 3) if df_supply["val1"].sum() != 0 else np.nan,
        round((df_supply["val2"] * df_supply["price2"]).sum() / df_supply["val2"].sum(), 3) if df_supply["val2"].sum() != 0 else np.nan,
        round((df_supply["val3"] * df_supply["price3"]).sum() / df_supply["val3"].sum(), 3) if df_supply["val3"].sum() != 0 else np.nan,
        round((df_supply["val4"] * df_supply["price4"]).sum() / df_supply["val4"].sum(), 3) if df_supply["val4"].sum() != 0 else np.nan
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
    # view_mode = st.sidebar.radio("Select View Mode:", ["Heatmaps", "Detailed Plots"], horizontal=False)
    view_mode = "Heatmaps"  # Force only Heatmaps mode, disables Detailed Plots

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
    st.markdown('The plots below show expenditure shares and prices (unit costs) split by demand and supply extracted from the pooled 2021-2023 MRIO data (the About tab). Average shares add up to one in the demand and supply columns respectively. Prices are nominal divided by real values.')


    #####################################
    #### detailed plots - top panel   ###
    #####################################


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


    #####################################
    ####    heatmaps - top panel   ###
    #####################################

    if view_mode == "Heatmaps":

        top_row_1 = st.container()

        with top_row_1:
            share_col, price_col = st.columns(2)

        with share_col:
            # Use already calculated mean_expenditures_demand and mean_expenditures_supply
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
            # Use already calculated mean_prices_demand and mean_prices_supply
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



    #### bottom panel ######

    st.markdown("### Elasticities")
    st.markdown("Elasticities are generated from a QUAIDS model using the pooled 2021-2023 MRIO data (see Notes). Results are split into Expenditure ($\\eta$) and Uncompensated Price ($\\epsilon^c$) elasticities. These are further split by demand and supply groups. Heatmaps show averages while Detailed Plots show the full distribution of the estimates.")



    #########################################
    ###   Detailed plots - bottom panel   ###
    #########################################   
    
    if view_mode == "Detailed Plots":

        eta_col1, eta_col2 = st.columns([1, 1])

        with eta_col1:
            st.plotly_chart(eta1, use_container_width=True, key="eta_demand")
        with eta_col2:
            st.plotly_chart(eta2, use_container_width=True, key="eta_supply")

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

    ###################################
    ###   Heatmaps - bottom panel   ###
    ###################################

    if view_mode == "Heatmaps":

        bot_row_1 = st.container()

        with bot_row_1:
            
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
            fig_eta_heat.update_layout(height=400, title="Expenditure/Income Elasticities")
            st.plotly_chart(fig_eta_heat)
            st.markdown("*Notes: $\\eta < 0$ is inferior good,  $0 < \\eta < 1$  is normal good, and $\\eta > 1$  is luxury good.*")


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
            
            st.markdown("*Notes: $\\epsilon_{ii} = 0$ is perfectly inelastic, $-1 <  \\epsilon_{ii} < 0$  is inelastic,  $\\epsilon_{ii}< -1$   is elastic,  $\\epsilon_{ii} \\rightarrow -\\infty$ is perfectly elastic. $\\epsilon_{ij} > 0$ are substitutes, $\\epsilon_{ij} < 0$ are complements.*")

    # Define helper functions
    def classify_price_elasticity(val):
        if val <= -6:
            return "perfectly elastic"
        elif -6 < val <= -1:
            return "relatively elastic"
        elif -1 < val < 0:
            return "relatively inelastic"
        elif val == 0:
            return "perfectly inelastic"
        else:
            return "undefined"

    def classify_income_elasticity(val):
        if val < 0:
            return "Inferior good"
        elif val < 1:
            return "Normal good with relatively inelastic income demand (necessity)"
        else:
            return "Normal good with relatively elastic income demand (luxury)"   

    def classify_cross_effects(cross_vals):
        max_val = max(cross_vals)
        min_val = min(cross_vals)
        if max_val > 0.2 and min_val < -0.2:
            return "substitution possible with other sectors (risk mitigation) but complementarity with other sectors also implies potential risk amplification"
        elif max_val > 0.2:
            return "substitution possible with other sectors (risk mitigation)"
        elif min_val < -0.2:
            return "strong complementarity with other sectors implies potential risk amplification"
        else:
            return "shows weak or neutral cross-price relationships (high risk sector)"

    # Good types mapping
    good_types = {
        1: "Intermediate Domestic",
        2: "Intermediate Foreign",
        3: "Final Domestic",
        4: "Final Foreign"
    }

    # Filter and prepare data
    # Use the correct country code column for summaries
    # Try both 'iso3' and 'country' columns for compatibility
    if 'iso3' in df.columns and 'name2' in df.columns:
        iso_to_name = dict(zip(df['iso3'].astype(str), df['name2'].astype(str)))
        country_code_col = 'iso3'
    elif 'country' in df.columns and 'name2' in df.columns:
        iso_to_name = dict(zip(df['country'].astype(str), df['name2'].astype(str)))
        country_code_col = 'country'
    else:
        iso_to_name = {}
        country_code_col = None

    elasticity_cols = ['iso3', 'type', 'ef_s1'] + \
        [f'eta_{i}' for i in range(1, 5)] + \
        [f'epsilon_{i}_{j}' for i in range(1, 5) for j in range(1, 5)]

    df_elasticities = df[elasticity_cols].copy()
    df_elasticities['type'] = df_elasticities['type'].astype(str).str.strip().str.lower()
    avg_elasticities = df_elasticities.groupby([country_code_col, 'type']).mean(numeric_only=True).reset_index()

    # Generate summaries
    summaries = {}

    for iso in avg_elasticities[country_code_col].unique():
        iso_str = str(iso)
        country_data = avg_elasticities[avg_elasticities[country_code_col] == iso]
        demand_row = country_data[country_data['type'] == 'demand']
        supply_row = country_data[country_data['type'] == 'supply']
        if demand_row.empty or supply_row.empty:
            continue

        summary_lines = [f"### Summary of results"]



        for i in range(1, 5):
            good = good_types[i]
            d_price = demand_row[f'epsilon_{i}_{i}'].values[0]
            d_income = demand_row[f'eta_{i}'].values[0]
            s_price = supply_row[f'epsilon_{i}_{i}'].values[0]
            s_income = supply_row[f'eta_{i}'].values[0]

            d_price_class = classify_price_elasticity(d_price)
            d_income_class = classify_income_elasticity(d_income)
            s_price_class = classify_price_elasticity(s_price)
            s_income_class = classify_income_elasticity(s_income)

            d_cross = [demand_row[f'epsilon_{i}_{j}'].values[0] for j in range(1, 5) if j != i]
            s_cross = [supply_row[f'epsilon_{i}_{j}'].values[0] for j in range(1, 5) if j != i]
            d_cross_desc = classify_cross_effects(d_cross)
            s_cross_desc = classify_cross_effects(s_cross)

            # Income shock impact
            if d_income < 0 and s_income < 0:
                income_impact = "Both demand and supply respond negatively to income increases."
            elif d_income >= 1 and s_income >= 1:
                income_impact = "Both demand and supply are highly responsive to income , income shocks will have strong effects."
            elif d_income < 1 and s_income < 1:
                income_impact = "Both demand and supply are inelastic to income, income shocks will have limited effects."
            elif d_income >= 1:
                income_impact = "Income shocks mainly affect demand as it is more income elastic than supply."
            elif s_income >= 1:
                income_impact = "Income shocks mainly affect supply as it is more income elastic than demand."
            else:
                income_impact = "Mixed income responsiveness between demand and supply."

            # Own price shock impact
            if d_price <= -6 and s_price <= -6:
                price_impact = (
                    "Both demand and supply are perfectly price elastic. Price shocks will lead to large quantity changes on both sides."
                )
            elif d_price <= -6:
                price_impact = (
                    "Demand is perfectly price elastic (highly responsive), while supply is less responsive. Price shocks will primarily impact demand quantities."
                )
            elif s_price <= -6:
                price_impact = (
                    "Supply is perfectly price elastic (highly responsive), while demand is less responsive. Price shocks will primarily impact supply quantities."
                )
            elif -6 < d_price <= -1 and -6 < s_price <= -1:
                price_impact = (
                    "Both demand and supply are relatively price elastic. Price shocks will significantly affect quantities on both sides."
                )
            elif -6 < d_price <= -1:
                price_impact = (
                    "Demand is relatively price elastic (responsive), while supply is less so. Price shocks will mainly affect demand quantities."
                )
            elif -6 < s_price <= -1:
                price_impact = (
                    "Supply is relatively price elastic (responsive), while demand is less so. Price shocks will mainly affect supply quantities."
                )
            elif -1 < d_price < 0 and -1 < s_price < 0:
                price_impact = (
                    "Both demand and supply are relatively price inelastic. Price shocks will have limited effects on quantities."
                )
            elif -1 < d_price < 0:
                price_impact = (
                    "Demand is relatively price inelastic (less responsive), while supply is more elastic. Price shocks will have limited impact on demand."
                )
            elif -1 < s_price < 0:
                price_impact = (
                    "Supply is relatively price inelastic (less responsive), while demand is more elastic. Price shocks will have limited impact on supply."
                )
            elif d_price == 0 and s_price == 0:
                price_impact = (
                    "Both demand and supply are perfectly price inelastic. Price shocks will not affect quantities on either side."
                )
            elif d_price == 0:
                price_impact = (
                    "Demand is perfectly price inelastic (no response), while supply is more responsive. Price shocks will not affect demand quantities."
                )
            elif s_price == 0:
                price_impact = (
                    "Supply is perfectly price inelastic (no response), while demand is more responsive. Price shocks will not affect supply quantities."
                )
            else:
                price_impact = (
                    "Demand and supply responsiveness is mixed or undefined. The impact of price shocks will vary depending on relative elasticities."
                )

            # Cross-price shock impact
            if "substitution" in d_cross_desc or "substitution" in s_cross_desc:
                cross_impact = "Cross-price elasticities indicate potential for substitution with other sectors, allowing for risk mitigation."
            elif "complementarity" in d_cross_desc or "complementarity" in s_cross_desc:
                cross_impact = "Cross-price elasticities indicate strong complementarity, potentially amplifying risk from shocks across related sectors."
            else:
                cross_impact = "Cross-price elasticities are weak or neutral, indicating limited risk transmission across sectors."



            summary_lines.append(
                f"- **{good}**\n"
                f"    - Demand: *{d_income_class}* (η = {d_income:.2f}), *{d_price_class}* to own prices (ε = {d_price:.2f}).\n"
                f"    - Supply: *{s_income_class}* (η = {s_income:.2f}), *{s_price_class}* to own prices (ε = {s_price:.2f}).\n"
                f"    - Impact of income shock: {income_impact}\n"
                f"    - Impact of price shock: {price_impact} {cross_impact}\n"
            )


            # driver = "demand" if d_price > s_price else "supply"

            # sector_importance = "high dependence" if d_price > 1 or s_price > 1 else "moderate to low dependence"

            # summary_lines.append(
            #    f"- **{good}**: Demand is {d_price_class} to prices (ε = {d_price:.2f}) and shows {d_income_class} (η = {d_income:.2f}). "
            #    f"Supply is {s_price_class} to prices (ε = {s_price:.2f}) and shows {s_income_class} (η = {s_income:.2f}). "
            #    f"A price shock is likely to be absorbed primarily through the *{driver} side*."
            #)

            #summary_lines.append(
            #    f"- **{good}**\n    - Demand: *{d_income_class}* (η = {d_income:.2f}), *{d_price_class}* to own prices (ε = {d_price:.2f}). Cross price elasticities indicate *{d_cross_desc}*.\n  - Supply: *{s_income_class}* (η = {s_income:.2f}), *{s_price_class}* to own prices (ε = {s_price:.2f}). Cross price elasticities indicate *{s_cross_desc}*. \n  - A price shock is likely to be absorbed through the *{driver} side*."
            #)

        #d_avg_price = demand_row[[f'epsilon_{i}_{i}' for i in range(1, 5)]].mean(axis=1).values[0]
        #s_avg_price = supply_row[[f'epsilon_{i}_{i}' for i in range(1, 5)]].mean(axis=1).values[0]

        #if d_avg_price > s_avg_price:
        #    shock_channel = "demand-side channels"
        #else:
        #    shock_channel = "supply-side channels"

        #d_avg_income = demand_row[[f'eta_{i}' for i in range(1, 5)]].mean(axis=1).values[0]
        #s_avg_income = supply_row[[f'eta_{i}' for i in range(1, 5)]].mean(axis=1).values[0]

        #if d_avg_price + d_avg_income > s_avg_price + s_avg_income:
        #    income_dynamic = "households"
        #else:
        #    income_dynamic = "firms"

        #summary_lines.append("")

        #summary_lines.append(
        #    f"**Overall**, "
            #f"**Overall**, demand and supply show average price elasticities of {d_avg_price:.2f} and {s_avg_price:.2f}, respectively. "
            #f"Average income elasticities are {d_avg_income:.2f} (demand) and {s_avg_income:.2f} (supply). "
        #    f"these results suggest that price shocks are most likely to propagate through *{shock_channel}*, requiring policies targeting *{income_dynamic}*."
        #)

        #summary_lines.append("")
        #summary_lines.append(
        #    f"**Overall**, market dynamics reflect average **demand elasticities** of ε = {d_avg_price:.2f}, η = {d_avg_income:.2f} and **supply elasticities** of ε = {s_avg_price:.2f}, η = {s_avg_income:.2f}. "
        #    f"This suggests that both price and income shocks will propagate through a mix of consumer and producer responses, with a slightly greater influence from the " + 
        #    ("*demand*" if (d_avg_price + d_avg_income) > (s_avg_price + s_avg_income) else "*supply*") + " side,"
        #    f" requiring policies targeted more towards *{income_dynamic}*."
        #)        

        #if d_avg_income > s_avg_income and d_avg_price <= s_avg_price:
        #    overall_msg = "Demand appears more responsive to income, while supply is more reactive to prices, indicating the need for coordinated policy attention across both consumers and producers."
        #elif d_avg_income > s_avg_income:
        #    overall_msg = "Demand show a stronger responsiveness to income changes, while price dynamics play a larger role on the supply side."
        #elif s_avg_income > d_avg_income:
        #    overall_msg = "Producers are more sensitive to both income and price changes, suggesting that production-side policies are needed."
        #else:
        #    overall_msg = "Elasticity patterns suggest a mixed or inconclusive response, and both demand and supply channels may be important depending on the type of shock."

        #summary_lines.append(f"**Overall**, {overall_msg}")

        country_name = iso_to_name.get(iso_str, iso_str)
        summaries[country_name] = "\n".join(summary_lines)

    # Streamlit interface
    # Use the country name from the sidebar selectbox for lookup
    if country in summaries:
        st.markdown(summaries[country])
    else:
        st.warning("Elasticity summary not available for the selected country.")
