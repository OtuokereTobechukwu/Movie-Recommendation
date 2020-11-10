import streamlit as st
import pandas as pd
from initial_approach import similar_movies
from alternate_approach import recommend_movies_by_plot

st.title('Movie recommendation Engine')
dataset = 'tmdb_5000_movies.csv'

@st.cache
def load_data():
    df = pd.read_csv(dataset)
    return df


data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text("Done!")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

# Radio buttons to select recommendation parameters
st.sidebar.subheader('Tuning parameters')
recommendation_list = ['By plot', 'By ratings']
rec_radio = st.sidebar.radio(label='Recommendation parameters', options=recommendation_list)

# A select box widget to select movies
st.subheader('Please select a movie to fetch similar movies')
movie = st.selectbox(label='Select a movie', options=data['title'])

# Creating recommendations based on radio button choice
if rec_radio == 'By plot':
    recommended_by_plot = recommend_movies_by_plot(movie)
    st.write(recommended_by_plot)
else:
    recommended_by_rating = similar_movies(movie)
    st.write(recommended_by_rating)

