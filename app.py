from flask import Flask, request, render_template
import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

app = Flask("__main__")
# Load the MovieLens datasets
movie_ratings = pd.read_csv("rating.csv")
movie_data = pd.read_csv("movie.csv")

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(movie_ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

# Build a collaborative filtering model
sim_options = {
    'name': 'cosine',
    'user_based': False
}
model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

@app.route('/')
def index():
    return render_template('index.html', recommendations=None)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    
    # Generate movie recommendations for the specified user
    recommendations = {}
    for movie_id in movie_data['movieId']:
        if movie_id not in movie_ratings['movieId'][movie_ratings['userId'] == user_id].values:
            predictions = model.predict(user_id, movie_id)
            recommendations[movie_id] = movie_data['title'][movie_data['movieId'] == movie_id].values[0]
    
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
