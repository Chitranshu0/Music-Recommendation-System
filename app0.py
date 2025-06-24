from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and data from 'model/' directory
train_df = pickle.load(open('model/musicrec_train_df.pkl', 'rb'))
tfidf = pickle.load(open('model/tfidf_vectorizer.pkl', 'rb'))
tfidf_vectors = pickle.load(open('model/tfidf_vectors.pkl', 'rb'))
tfidf_similarity = pickle.load(open('model/tfidf_similarity.pkl', 'rb'))

def recommend(song_title, top_n=5):
    try:
        idx = train_df[train_df['title'] == song_title].index[0]
        sim_scores = list(enumerate(tfidf_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        recommended = [train_df.iloc[i[0]].title for i in sim_scores]
        return recommended
    except IndexError:
        return ["No recommendations found. Try a different song."]

@app.route('/', methods=['GET'])
def index():
    songs = train_df['title'].tolist()
    return render_template('index.html', songs=songs)

@app.route('/recommend', methods=['POST'])
def recommend_route():
    song_name = request.form.get('song_name')
    recommendations = recommend(song_name)
    return render_template('recommend.html', selected_song=song_name, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)