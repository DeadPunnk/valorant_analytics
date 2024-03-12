#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
#import matplotlib.pyplot as plt
#import seaborn as sns
#import altair as alt
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

st.set_page_config(page_title="Valorant Analytics", page_icon="🎮", initial_sidebar_state="expanded", layout='wide')

st.title("E-sports Analytics Dashboard")
#st.subheader('email: joaopedroqnd@gmail.com, telegran: @Jotinha07, linkedin: https://www.linkedin.com/in/joaopedroborges98/')

#@st.cache
def load_and_prep_players():
    dfplayers = pd.read_csv('archive/stats.csv')
    tb_data = dfplayers.loc[dfplayers['org'] == 'TB']
    dfplayers = tb_data
    #dfplayers['KDA'] = str(round(((dfplayers['kills'] + dfplayers['assists'])/dfplayers['deaths']), 2))
    #dfplayers = dfplayers.set_index(dfplayers['date'])


    
    return dfplayers

def load_and_prep_teams():
    dfcomp = pd.read_csv('archive/first_strike.csv')
    #dfplayers['KDA'] = str(round(((dfplayers['kills'] + dfplayers['assists'])/dfplayers['deaths']), 2))
    #dfteams = dfteams.set_index(dfplayers['date'])
    return dfcomp

dfplayers = load_and_prep_players()
dfcomp = load_and_prep_teams() 
#dfplayers22 = pd.read_csv('playersst_22.csv')


def convert_df(df):

   return df.to_csv(index=False).encode('utf-8')


def color_surplusvalue(val):
    if str(val) == '0':
        color = 'azure'
    elif str(val)[0] == '-':
        color = 'lightpink'
    else:
        color = 'lightgreen'
    return 'background-color: %s' % color


heading_properties = [('font-size', '16px'),('text-align', 'center'),
                      ('color', 'white'),  ('font-weight', 'bold'),
                      ('background', 'green'),('border', '1.2px solid')]

cell_properties = [('font-size', '16px'),('text-align', 'center'), ('color', 'black'), ('text-align', 'center'), ('font-weight', 'bold')]


dfstyle = [{"selector": "th", "props": heading_properties},
               {"selector": "td", "props": cell_properties}]


#tab_player, tab_team = st.tabs(["Player", "Time"])


cols = ['date','teamname', 'position', 'kills', 'deaths', 'assists', 'KDA', 'totalgold', 'total cs']


tab2, tab3 = st.tabs(['Players', 'Detalhe'])




def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    #modify = st.checkbox("Add filters", key = 123)

    #if not modify:
    #    return df

    df = df.copy()



    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filtrar tabela", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Valores {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Valores {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring ou regex {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df





#cols2 = ['teamname', 'split', 'result', 'teamkills', 'teamdeaths', 'totalgold', 'towers', 'dragons', 'barons']

#with tab1:

#    st.write(f'''
#         ##### <div style="text-align: center">First Strike Europa 2020<span style="color:blue">
#         ''', unsafe_allow_html=True)
#

with tab2:

    c1, c2 = st.columns((1,1))


    c1.write(f'''
         ##### <div style="text-align: center">Kills<span style="color:blue">
         ''', unsafe_allow_html=True)

    chart_data_bar = dfplayers
    c1.bar_chart(data = chart_data_bar, x = 'player', y = 'total_kills')

    c2.write(f'''
         ##### <div style="text-align: center">Deaths<span style="color:blue">
         ''', unsafe_allow_html=True)    


    chart_data_bar2 = dfplayers
    c2.bar_chart(data = chart_data_bar2, x = 'player', y = 'total_deaths')

with tab3:

    st.write(f'''
         ##### <div style="text-align: center">First Strike Europa 2020<span style="color:blue">
         ''', unsafe_allow_html=True)

    comp_selected = filter_dataframe(dfcomp)
    st.dataframe(comp_selected)

    csv = convert_df(comp_selected)


    st.download_button("Download", csv, "cblol_detalhe.csv", "text/csv", key='download-csv')

