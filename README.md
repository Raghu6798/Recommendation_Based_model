# Recommendation System for User Posts

## Overview

This project implements a recommendation system that suggests posts to users based on their interactions and preferences. The system uses a combination of Natural Language Processing (NLP) and machine learning techniques to provide personalized recommendations. The Flask API exposes endpoints where users can retrieve recommended posts based on their username, category, and mood.

## Features

- **User-specific recommendations**: The system generates recommendations based on a user’s previous interactions with posts, considering factors like upvotes, comments, and shares.
- **Category-based recommendations**: Users can filter recommendations by category.
- **Mood-based recommendations**: Recommendations can be filtered based on the user’s mood.
- **Hyperparameter optimization**: Uses Optuna to optimize the weights for different features in the recommendation algorithm.
- **Web API**: Provides 3 API endpoints to fetch recommendations.

## Technology Stack

- **Flask**: Web framework for creating API endpoints.
- **Pandas**: Data processing and manipulation.
- **scikit-learn**: For text vectorization, feature scaling, and similarity calculations.
- **Optuna**: Hyperparameter optimization for personalized recommendations.
- **NumPy**: For numerical operations and array manipulation.

## Project Structure

```plaintext
.
├── app.py                # Main Flask application with API endpoints
├── filtered_df.csv       # CSV file containing user post data
├── requirements.txt      # Project dependencies
├── templates/
│   ├── index.html        # User input form for username, category, and mood
│   ├── output.html       # Displays recommendations to the user
├── README.md             # This file

