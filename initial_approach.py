# Building a recommendation system
# Initial approach based on vote_average, runtime and popularity.

# Importing needed libraries
import pandas as pd


# Importing dataset
movie_data = pd.read_csv("tmdb_5000_movies.csv")
# credits_data = pd.read_csv("tmdb_5000_credits.csv")

# ---------------- Data pre-processing -------------- #
# Select our columns of interest
cols_of_interest = ['id', 'title', 'original_language', 'popularity',
                    'runtime', 'vote_average', 'vote_count', 'revenue']
# Create a new dataset with the selected columns above
new_data = movie_data[cols_of_interest]

# First we select more granular columns and then expand the functionality
rec_cols = ['title', 'vote_average', 'vote_count']
simple_rec_data = new_data[rec_cols].set_index('title')

# ----------------- Building the recommendation system ------------------ #
# Creating a matrix by pivoting the data
matrix = new_data.pivot_table(columns='title', values=['vote_average', 'runtime', 'popularity'])

# A function to recommend similar movies
def similar_movies(name):
    # select features of the movie from the matrix
    movie_features = matrix[name]
    # correlate selected features with matrix to expose similarities
    movie_like = matrix.corrwith(movie_features)
    # convert to dataframe
    corr_to_movie = pd.DataFrame(movie_like, columns=['Correlation'], index=None)
    corr_to_movie.dropna(inplace=True)
    # Join the vote_count to the newly created dataframe and recommend movies that have above 3000 user votes
    corr_to_movie = corr_to_movie.join(simple_rec_data['vote_count']).sort_values(by='vote_count', ascending=False)
    like_movies = corr_to_movie[corr_to_movie['vote_count'] > 3000].sort_values(by='Correlation',
                                                                                ascending=False).head(10)

    return like_movies


# Testing the recommendation system
print(similar_movies('Iron Man 3'))

# Results

# Iron Man 3                                1.000000        8806
# Hansel & Gretel: Witch Hunters            1.000000        3239
# Amélie                                    0.999998        3310
# Captain America: The First Avenger        0.999994        7047
# The Lord of the Rings: The Two Towers     0.999994        7487
# The Usual Suspects                        0.999989        3254
# Shutter Island                            0.999976        6336
# Star Trek Into Darkness                   0.999973        4418
# The Fault in Our Stars                    0.999971        3759
# The Hobbit: The Desolation of Smaug       0.999938        4524