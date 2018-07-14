import numpy as np
import pandas as pd

movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')
ratings.drop(['timestamp'], axis=1, inplace=True)
# print(movies[movies['movieId']==50])
# print(movies.head())
# print(movies[movies['title'] == '10 Things I Hate About You (1999)'])
def replace_name(x):
    return movies[movies['movieId']==x].title.values[0]

ratings.movieId = ratings.movieId.map(replace_name)
# print(ratings.head())

M = ratings.pivot_table(index=['userId'], columns=['movieId'], values='rating')
print(M.shape)
# print(M)
def pearson(s1, s2):
    s1_c = s1 - s1.mean()
    s2_c = s2 - s2.mean()
    return np.sum(s1_c * s2_c)/np.sqrt(np.sum(s1_c ** 2) * np.sum(s2_c ** 2))
p = pearson(M['\'burbs, The (1989)'], M['10 Things I Hate About You (1999)'])
# print(p)
def get_recs(movie_name, M, num):
    reviews = []
    for title in M.columns:
        if title == movie_name:
            continue
        cor = pearson(M[movie_name], M[title])
        if np.isnan(cor):
            continue
        else:
            reviews.append((title, cor))
    reviews.sort(key=lambda tup: tup[1], reverse=True)
    return reviews[:num]

# recs = get_recs('Clerks (1994)', M, 10)
# print(recs)
# anti_recs = get_recs('Clerks (1994)', M, 8551)
# print(anti_recs[-10:])