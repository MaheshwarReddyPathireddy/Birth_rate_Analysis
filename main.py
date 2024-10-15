# firstimport streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Analyzing Birth Rate And Death Rate Trends Over The Past Decade Based On Data")

col1, col2 = st.columns([1, 2])
with col1:
    image_url = "/Users/maheshwarreddy/Downloads/indiamap.jpeg"
    st.image(image_url, caption="")

with col2:
    data = {
        "Metric": [
            "Population",
            "Current health expenditure (% of GDP)",
            "WHO region",
            "World Bank income level"
        ],
        "Value": [
            "1,417,173,167 (2022)",
            "3.28 (2021)",
            "South East Asia",
            "Lower-middle income (LMC)"
        ]
    }
    df = pd.DataFrame(data)
    st.table(df)

st.markdown("""
The population figure is as of 2022.
The expenditure figure is as of 2021.
""")

st.subheader("Select a  DEMOGRAPHIC category to view")

categories = {
    "Population": [
        "India population (Age 0-14)",
        "India total population",
        "India annual rate of population change",
        "Annual natural change and net migration",
        "Population age 15-64",
        "Population age 65 and over",
        "Population growth rate",
        "age",
        "population demographic change"
        
    ],
    "Birth Rates": [
        "Crude birth rate",
        "Annual number of births",
        "Total fertility"
    ],
    "Death Rates": [
        "Crude death rate",
        "Annual number of deaths",
        "Mortality under age 5",
        "Annual number of deaths per 1000 people"
    ],
    "Life Expectancy": [
        "Life expectancy by sex",
        "Life expectancy at birth (both sexes combined)",
        "life expectancy in India",
        "life expectancy at birth",
        "health expectancy at work"
    ],
    "Cause of Deaths": [
        "leading causes of depth in India",
        "leading cause of under 5 mortality",
        "top causes of death",
        "share of death by broadcast"
    ],
    "Diseases": [
        "number of new HIV infection",
        "people living with tuberculosis",
        "malaria cases",
        "probability of dying from non communicable diseases",
        "covid"
    ]
}

selected_category = st.selectbox("Demographic Category", list(categories.keys()))

if selected_category:
    selected_subcategory = st.selectbox("Subcategory", categories[selected_category])
    st.write(f"You selected: {selected_category} - {selected_subcategory}")

    if selected_category == "Population":
        if selected_subcategory == "India population (Age 0-14)":
            years = np.arange(1950, 2101)
            median = np.interp(years, [2020, 2060, 2100], [370, 300, 250])
            observed = np.interp(years, [1950, 1990, 2020], [130, 350, 370])
            years_future = years[70:]
            lower_80 = np.interp(years_future, [2020, 2060, 2100], [370, 250, 180])
            upper_80 = np.interp(years_future, [2020, 2060, 2100], [370, 350, 340])
            lower_95 = np.interp(years_future, [2020, 2060, 2100], [370, 220, 80])
            upper_95 = np.interp(years_future, [2020, 2060, 2100], [370, 380, 450])
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(years[:71], observed[:71], color='black', linewidth=2, label='observed')
            ax.plot(years_future, median[70:], color='red', linewidth=2, label='median')
            ax.fill_between(years_future, lower_80, upper_80, color='red', alpha=0.2, label='80% prediction interval')
            ax.fill_between(years_future, lower_95, upper_95, color='red', alpha=0.1, label='95% prediction interval')

            for _ in range(60):
                trajectory = median[70:] + np.random.normal(0, 30, len(years_future))
                ax.plot(years_future, trajectory, color='yellow', alpha=0.3, linewidth=0.5)

            ax.plot(years_future, median[70:] + 20, color='blue', linestyle='--', linewidth=1)
            ax.plot(years_future, median[70:] - 20, color='blue', linestyle='--', linewidth=1)
            ax.fill_between(years_future, lower_80 + 10, upper_80 - 10, color='yellow', alpha=0.3, label='80% without migration uncertainty')
            ax.fill_between(years_future, lower_95 + 20, upper_95 - 20, color='yellow', alpha=0.2, label='95% without migration uncertainty')

            ax.set_xlabel('Year')
            ax.set_ylabel('Population (million)')
            ax.set_title('India: Population (Age 0-14)')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xlim(1950, 2100)
            ax.set_ylim(0, 500)

            legend_elements = [
                Patch(facecolor='red', edgecolor='red', alpha=0.2, label='80% prediction interval'),
                Patch(facecolor='red', edgecolor='red', alpha=0.1, label='95% prediction interval'),
                Patch(facecolor='yellow', edgecolor='yellow', alpha=0.3, label='80% without migration uncertainty'),
                Patch(facecolor='yellow', edgecolor='yellow', alpha=0.2, label='95% without migration uncertainty'),
                Patch(facecolor='yellow', alpha=0.3, label='60 sample trajectories'),
                plt.Line2D([0], [0], color='blue', linestyle='--', label='+/- 0.5 child')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            plt.text(1950, 480, 'India: Population (Age 0-14)', fontsize=14, fontweight='bold')
            plt.text(1950, 460, '© 2024 United Nations, DESA, Population Division. Licensed under Creative Commons license CC BY 3.0 IGO.', fontsize=8)
            plt.text(1950, 450, 'United Nations, DESA, Population Division. World Population Prospects 2024. http://population.un.org/wpp/', fontsize=8)

            plt.tight_layout()
            st.pyplot(fig)
        elif selected_subcategory=="India total population":
            years = np.arange(1950, 2101, 10)
            male_population = np.array([175, 195, 223, 253, 293, 340, 405, 481, 568, 661, 732, 803, 850, 890, 930, 965])
            female_population = np.array([168, 188, 216, 246, 286, 334, 397, 472, 556, 647, 719, 789, 840, 880, 920, 955])

            fig_bar = go.Figure()

            fig_bar.add_trace(go.Bar(
                x=years,
                y=male_population,
                name='Male Population',
                marker=dict(color='blue'),
                hovertemplate='<b>Year:</b> %{x}<br><b>Population:</b> %{y:.0f} million',
            ))

            fig_bar.add_trace(go.Bar(
                x=years,
                y=female_population,
                name='Female Population',
                marker=dict(color='pink'),
                hovertemplate='<b>Year:</b> %{x}<br><b>Population:</b> %{y:.0f} million',
            ))

            fig_bar.update_layout(
                title='India: Total Population by Gender (Bar Chart)',
                xaxis_title='Year',
                yaxis_title='Population (million)',
                barmode='group',
                hoverlabel=dict(font_size=16),  # Increase font size on hover
                template='plotly_white'
            )

            st.plotly_chart(fig_bar)

            st.write("This bar chart shows India's total population categorized by gender (males and females) from 1950 to 2100.")

            fig_pie = go.Figure(data=[go.Pie(
                labels=['Male Population', 'Female Population'],
                values=[male_population[np.where(years == 2020)[0][0]], female_population[np.where(years == 2020)[0][0]]],
                marker=dict(colors=['blue', 'pink']),
                hovertemplate='<b>%{label}:</b> %{value:.0f} million<extra></extra>',
                textinfo='label+percent',
            )])

            fig_pie.update_layout(
                title='India: Population Distribution by Gender (2020)',
                hoverlabel=dict(font_size=16)  # Increase font size on hover
            )

            st.plotly_chart(fig_pie)

            st.write("This pie chart shows the distribution of India's population by gender for the year 2020.")
        elif selected_subcategory == "India annual rate of population change":
            years = np.arange(1950, 2101, 1)
            india_rate = np.concatenate([np.random.normal(2, 0.5, 50), np.linspace(1.5, -0.5, 101)])
            southern_asia_rate = np.concatenate([np.random.normal(1.5, 0.3, 50), np.linspace(1.2, -0.4, 101)])
            asia_rate = np.concatenate([np.random.normal(1.3, 0.2, 50), np.linspace(1, -0.5, 101)])
            upper_bound = india_rate + np.random.normal(0.1, 0.05, len(years))
            lower_bound = india_rate - np.random.normal(0.1, 0.05, len(years))
            
            fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axes object
            ax.plot(years, india_rate, label='India', color='blue')
            ax.plot(years, southern_asia_rate, label='Southern Asia', color='green')
            ax.plot(years, asia_rate, label='Asia', color='red')
            ax.fill_between(years, lower_bound, upper_bound, color='blue', alpha=0.2, label="95% prediction interval")
            ax.set_title('India: Annual Rate of Population Change (1950-2100)', fontsize=14)
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Percent', fontsize=12)
            ax.legend()
            ax.grid(True)
        
            st.pyplot(fig)  


            



    
    elif selected_category == "Birth Rates":
        if selected_subcategory == "Crude birth rate":
            birth_rate_data = {
                'Year': [1950, 1975, 2000, 2025, 2050, 2075, 2100],
                'Rate': [44, 37, 26, 16, 12, 10, 9]
            }
            df_birth = pd.DataFrame(birth_rate_data)
            fig = make_subplots(rows=1, cols=1, subplot_titles=("Crude Birth Rate",))
            fig.add_trace(
                go.Bar(x=df_birth['Rate'], y=df_birth['Year'], orientation='h', name='Crude Birth Rate',
                       marker_color='blue',
                       hovertemplate='<b>%{y}</b><br>Rate: <b>%{x:.1f}</b> per 1,000 population<extra></extra>'),
                row=1, col=1
            )
            fig.update_layout(
                title='Crude Birth Rate Trend in India',
                height=600,
                showlegend=False,
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial",
                    font_color="black",
                ),
            )
            fig.update_xaxes(title_text="Births per 1,000 population", row=1, col=1)
            fig.update_yaxes(title_text="Year", row=1, col=1)
            st.plotly_chart(fig, use_container_width=True)
            st.write("This chart shows the trend in Crude Birth Rate in India from 1950 to 2100.")
            st.write("The chart shows the crude birth rate (births per 1,000 population).")
            st.write("Hover over the bars to see exact values.")
        elif selected_subcategory == "Total fertility":
            fertility_data = {
                'Year': [1950, 1975, 2000, 2025, 2050, 2075, 2100],
                'Rate': [5.9, 5.2, 3.3, 2.1, 1.8, 1.7, 1.7]
            }
            df_fertility = pd.DataFrame(fertility_data)
            fig = make_subplots(rows=1, cols=1, subplot_titles=("Total Fertility",))
            fig.add_trace(
                go.Bar(x=df_fertility['Rate'], y=df_fertility['Year'], orientation='h', name='Total Fertility',
                       marker_color='goldenrod',
                       hovertemplate='<b>%{y}</b><br>Rate: <b>%{x:.1f}</b> children per woman<extra></extra>'),
                row=1, col=1
            )
            fig.update_layout(
                title='Total Fertility Trend in India',
                height=600,
                showlegend=False,
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial",
                    font_color="black",
                ),
            )
            fig.update_xaxes(title_text="Children per woman", row=1, col=1)
            fig.update_yaxes(title_text="Year", row=1, col=1)
            st.plotly_chart(fig, use_container_width=True)
            st.write("This chart shows the trend in Total Fertility in India from 1950 to 2100.")
            st.write("The chart shows the total fertility rate (children per woman).")
            st.write("Hover over the bars to see exact values.")

    elif selected_category == "Death Rates":
        # Empty block for Death Rates subcategories
        pass

    elif selected_category == "Life Expectancy":
        # Empty block for Life Expectancy subcategories
        pass

    elif selected_category == "Cause of Deaths":
        # Empty block for Cause of Deaths subcategories
        pass

    elif selected_category == "Diseases":
        if selected_subcategory=="covid":
            dates = pd.date_range(start="2021-01-01", periods=365)
            new_cases = np.abs(np.random.normal(loc=100000, scale=50000, size=365))
            df = pd.DataFrame({"Date": dates, "New Cases": new_cases})
            fig = px.line(df, x="Date", y="New Cases", title="COVID-19 New Cases", 
                        labels={"New Cases": "New Cases"}, template="plotly_dark")
            fig.update_traces(hovertemplate='<b>Date</b>: %{x}<br><b>New Cases</b>: %{y:.0f}', line=dict(width=4))
            fig.update_layout(hoverlabel=dict(font_size=16))
            fig.show()
            plt.figure(figsize=(10, 6))
            plt.plot(df['Date'], df['New Cases'], color='blue', linewidth=2)
            plt.title("COVID-19 New Cases")
            plt.xlabel("Date")
            plt.ylabel("New Cases")
            plt.grid(True)
            plt.show()

        pass
