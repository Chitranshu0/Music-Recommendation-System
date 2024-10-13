from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

new_df = pickle.load(open('model/musicrec.pkl', 'rb'))
similarity = pickle.load(open('model/similarities.pkl', 'rb'))

def recommend(music):
    try:
        music_index = new_df[new_df['title'] == music].index[0]
        distances = similarity[music_index]
        
        music_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        recommended_songs = [new_df.iloc[i[0]].title for i in music_list]
        
        return recommended_songs
    except IndexError:
        return ["No recommendations found. Try a different song."]


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend_route():
    song_name = request.form.get('song_name')
    recommendations = recommend(song_name)
    return render_template('recommend.html', selected_song=song_name, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
