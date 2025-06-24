from flask import Flask, render_template, request
import pickle, pandas

app = Flask(__name__)

# Load model and data from 'model/' directory
train_df = pickle.load(open('model/musicrec_train_df.pkl', 'rb'))
tfidf_similarity = pickle.load(open('model/tfidf_similarity.pkl', 'rb'))

song_df= pandas.read_csv("data/ex.csv")

def recommend(song_title, top_n=5):
    try:
        idx = train_df[train_df['title'] == song_title].index[0]
        sim_scores = list(enumerate(tfidf_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        recommended = [train_df.iloc[i[0]].title for i in sim_scores]
        return recommended
    except IndexError:
        return ["No recommendations found. Try a different song."]
    
def get_song_data(song_titles):
    songs_data = []
    try:
        for i, song in enumerate(song_titles):
            song_data = song_df[song_df["Song-Name"]==song]
            songs_data.append([i+1,
                song_data["Song-Name"].loc[song_data.index[0]],
                song_data["Singer/Artists"].loc[song_data.index[0]],
                song_data["Album/Movie"].loc[song_data.index[0]],
                song_data["User-Rating"].loc[song_data.index[0]]])
    except Exception:
        songs_data = [["NaN", "None", "None", "None", "None"]]
        
    return songs_data

@app.route('/', methods=['GET'])
def index():
    songs = train_df['title'].tolist()
    song_title = songs[0]
    recommended_titles = recommend(song_title)
    song_data = get_song_data(recommended_titles)
    return render_template('index.html', songs=songs, song_title=song_title, recommended_songs=song_data)

@app.route('/recommend', methods=['POST'])
def recommend_route():
    songs = train_df['title'].tolist()
    song_title = request.form.get('song_name')
    recommended_titles = recommend(song_title)
    song_data = get_song_data(recommended_titles)
    return render_template('index.html', songs=songs, song_title=song_title, recommended_songs=song_data)

if __name__ == '__main__':
    app.run(debug=True)
