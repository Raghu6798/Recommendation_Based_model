# Recommendation System for User Posts

## Overview

This project implements a recommendation system that suggests posts to users based on their interactions and preferences. The system uses a combination of Natural Language Processing (NLP) and machine learning techniques to provide personalized recommendations. The Flask API exposes endpoints where users can retrieve recommended posts based on their username, category, and mood.

## Features

- **User-specific recommendations**: The system generates recommendations based on a user’s previous interactions with posts, considering factors like upvotes, comments, and shares.
- **Category-based recommendations**: Users can filter recommendations by category.
- **Mood-based recommendations**: Recommendations can be filtered based on the user’s mood.
- **Hyperparameter optimization**: Uses Optuna to optimize the weights for different features in the recommendation algorithm.

## Technology Stack

- **Flask**: Web framework for creating API endpoints.
- **Pandas**: Data processing and manipulation.
- **scikit-learn**: For text vectorization, feature scaling, and similarity calculations.
- **Optuna**: Hyperparameter optimization for personalized recommendations.
- **NumPy**: For numerical operations and array manipulation.

## Project Structure

``` 
├── app.py                # Main Flask application with API endpoints
├── filtered_df.csv       # CSV file containing user post data
├── requirements.txt      # Project dependencies
├── templates/
│   ├── index.html        # User input form for username, category, and mood
│   ├── output.html       # Displays recommendations to the user
├── README.md             # This file

## Setup Instructions

Follow these steps to set up and run the project on your local machine:

### 1. Clone the Repository

First, clone the repository to your local machine:

```
git clone https://github.com/Raghu6798/Recommendation_Based_model.git
cd Source_file.py

##2. Set Up Python Virtual Environment
It is highly recommended to set up a Python virtual environment to isolate the dependencies for this project.


For Windows:
python -m venv venv

2. Set Up Python Virtual Environment
It is highly recommended to set up a Python virtual environment to isolate the dependencies for this project.

For macOS/Linux:
bash
Copy code
python3 -m venv venv

3. Activate the Virtual Environment
For Windows:
```
Copy code
.\venv\Scripts\activate
For macOS/Linux:
```
Copy code
source venv/bin/activate

4. Install Dependencies
Once the virtual environment is activated, install the project dependencies using the following command:

```
Copy code
pip install -r requirements.txt
5. Run the Flask Application
To run the application, use the following command:

```
Copy code
python app.py
The Flask application will start running on http://localhost:5000/ by default.

```
6. Access the Web Interface
Once the Flask server is running, you can access the web interface and interact with the recommendation system by visiting the following URL in your browser:

```
Copy code
http://localhost:5000
