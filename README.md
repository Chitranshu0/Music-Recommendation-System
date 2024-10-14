# Music Recommendation System 🎶

## Overview
This project is a music recommendation system using machine learning. It recommends similar songs based on user input. Users can easily enter a song in the Flask web app and receive personalized recommendations.

## Tech Stack
- **Python**: For data processing (pandas, scikit-learn)
- **Flask**: Web framework for user interface
- **CountVectorizer**: Converts song features into vectors
- **Cosine Similarity**: Finds and recommends similar songs

## Features
- **Data Cleaning**: Removes duplicates and missing values.
- **Tag-Based Matching**: Uses a combination of artists, genres, and ratings for recommendations.
- **Web Interface**: Flask-based form where users input a song name.

## Usage
1. Clone the repository.
2. Install libraries: `pip install -r requirements.txt`
3. Run `app.py` and access at `http://127.0.0.1:5000`.

## License
This project is licensed under the MIT License.
