from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np
import optuna

app = Flask(__name__)

# Load the filtered_df CSV data
filtered_df = pd.read_csv(r"C:\Users\Raghu\Downloads\Recommender_ML\Data\filtered_df.csv")

# Preprocess the data: Combine text columns and scale numerical features
filtered_df['combined_text'] = (
    filtered_df['title'] + ' ' +
    filtered_df['description'] + ' ' +
    filtered_df['category_name'] + ' ' +
    filtered_df['emotions_conveyed'].astype(str)
)

# Vectorize the combined text using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_df['combined_text'])

# Normalize numerical features
numerical_columns = ['upvote_count', 'view_count', 'comment_count', 'rating_count', 'average_rating', 
                     'following', 'exit_count', 'post_count', 'share_count', 
                     'following_count', 'follower_count', 'upvoted', 'bookmarked']
scaler = StandardScaler()
filtered_df[numerical_columns] = scaler.fit_transform(filtered_df[numerical_columns])

# Function to build user profile
def build_user_profile(user, df, weights):
    df = df.reset_index(drop=True)
    user_posts = df[df['username'] == user]
    weighted_vectors = []
    for index, row in user_posts.iterrows():
        interaction_weight = (
            (row['upvote_count'] * weights[0] + row['comment_count'] * weights[1] + row['view_count'] * weights[2])
            + (row['rating_count'] * row['average_rating'] * weights[3])
            + (row['post_count'] * weights[4] + row['share_count'] * weights[5])
        )
        post_vector = tfidf_matrix[index].toarray()
        weighted_vectors.append(post_vector.flatten() * interaction_weight)

    user_profile = np.mean(weighted_vectors, axis=0) if weighted_vectors else np.zeros(tfidf_matrix.shape[1])
    return user_profile

# Function to get recommendations for a user
def get_recommendations_for_user(user, category, mood, top_n, weights):
    if user not in filtered_df['username'].unique():
        category_filtered = filtered_df[filtered_df['category_name'].str.contains(category, case=False, na=False)]
        mood_filtered = category_filtered[category_filtered['emotions_conveyed'].str.contains(mood, case=False, na=False)]
        recommendations_df = mood_filtered if not mood_filtered.empty else category_filtered
        return recommendations_df['title'].head(top_n).tolist()
    else:
        user_profile = build_user_profile(user, filtered_df, weights)
        cosine_similarities = cosine_similarity(user_profile.reshape(1, -1), tfidf_matrix).flatten()
        recommended_indices = cosine_similarities.argsort()[-top_n:][::-1]

        recommendations = []
        for idx in recommended_indices:
            if filtered_df.loc[idx, 'username'] != user:
                recommendations.append(filtered_df.loc[idx, 'title'])

        return recommendations

# Define the objective function for Optuna optimization
def optuna_objective(trial, username, category, mood, yhat):
    weights = [trial.suggest_float(f"weight_{i}", 0.0, 10.0) for i in range(6)]
    recommendations = get_recommendations_for_user(username, category, mood, len(yhat), weights)
    mae = calculate_mae(recommendations, yhat)
    rmse = calculate_rmse(recommendations, yhat)
    return mae + rmse

# Calculate MAE (Mean Absolute Error)
def calculate_mae(predictions, ground_truth):
    prediction_set = set(predictions)
    ground_truth_set = set(ground_truth)
    errors = len(ground_truth_set) - len(prediction_set.intersection(ground_truth_set))
    mae = abs(errors) / len(ground_truth_set) if ground_truth_set else 0
    return mae

# Calculate RMSE (Root Mean Square Error)
def calculate_rmse(predictions, ground_truth):
    prediction_set = set(predictions)
    ground_truth_set = set(ground_truth)
    errors = len(ground_truth_set) - len(prediction_set.intersection(ground_truth_set))
    rmse = np.sqrt(errors ** 2 / len(ground_truth_set)) if ground_truth_set else 0
    return rmse

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    username = request.form['username']
    category = request.form['category']
    mood = request.form['mood']
    
    # Define yhat (ground truth titles) based on the user's posts
    yhat = filtered_df[filtered_df['username'] == username]['title'].tolist() if username in filtered_df['username'].unique() else []
    
    # Run Optuna optimization to get the best weights
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: optuna_objective(trial, username, category, mood, yhat), n_trials=50)
    
    # Get optimized weights
    optimized_weights = [study.best_params[f"weight_{i}"] for i in range(6)]
    
    # Get recommendations based on optimized weights
    recommendations = get_recommendations_for_user(username, category, mood, len(yhat) or 5, optimized_weights)

    # Pass recommendations to Output.html
    return render_template('Output.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
