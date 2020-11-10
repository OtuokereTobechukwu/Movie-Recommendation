# Building a content based recommendation system hinged on the plot of the movie

# Importing needed libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Importing dataset
movie_data = pd.read_csv("tmdb_5000_movies.csv")
# credits_data = pd.read_csv("tmdb_5000_credits.csv")

cols_of_interest = ['id', 'title', 'overview']
movies = movie_data[cols_of_interest].copy()

tfidf = TfidfVectorizer(stop_words='english')
movies['overview'] = movies['overview'].fillna('')

# implement the matrix by using the fit_transform method
overview_matrix = tfidf.fit_transform(movies['overview'])
similarity_matrix = linear_kernel(overview_matrix, overview_matrix)

mapping = pd.Series(movies.index, index=movies['title'])

# A function to recommend movies
def recommend_movies_by_plot(movie):
    movie_index = mapping[movie]
    # Comparing with other movies to find similarities
    similarity_score = list(enumerate(similarity_matrix[movie_index]))
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    similarity_score = similarity_score[1:10]
    # Return movie names
    movie_indices = [i[0] for i in similarity_score]

    return (
        movies['title'].iloc[movie_indices]
    )


print(recommend_movies_by_plot('Iron Man 3'))

#Results

# 79                   Iron Man 2
# 1868         Cradle 2 the Grave
# 68                     Iron Man
# 1664              Dead Man Down
# 590                   The Siege
# 2193       Secret in Their Eyes
# 47      Star Trek Into Darkness
# 2044         The Little Vampire
# 7       Avengers: Age of Ultron