from flask import Flask, request, jsonify
import praw
import csv
import pandas as pd
from wordcloud import STOPWORDS
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import iplot
from flask_cors import CORS

from collections import defaultdict
from plotly import tools
import plotly.io as pio

import spacy
import json
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import en_core_web_sm

nlp = en_core_web_sm.load()


import pandas as pd
from googleapiclient.discovery import build
import io
import base64


# nlp = en_core_web_sm.load()

app = Flask(__name__)
CORS(app, origins="http://localhost:3000")

def process_string(input_string):

    ## any preprocessing needed
    processed_string = input_string
    return processed_string
## custom function for ngram generation ##

def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

## custom function for horizontal bar chart ##
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace

def summarize(text, per):
    nlp = spacy.load('en_core_web_sm')
    doc= nlp(text)
    tokens=[token.text for token in doc]
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    select_length=int(len(sentence_tokens)*per)
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
    final_summary=[word.text for word in summary]
    summary=''.join(final_summary)
    return summary 

@app.route('/searchYouTube', methods=['POST'])
def process_string_route():
    # Get the input string from the request data
    input_string = request.json.get('input_string')
    print("input_string: ")
    print(input_string)

    if not input_string:
        return jsonify({'error': 'Input string is missing'}), 400

    # processed_string = process_string(input_string)
    # api_key = "AIzaSyAWi-L9FRX0R27eBp3Iy7QfBao_XdijQto"

    # # Initialize YouTube Data API client
    # youtube = build('youtube', 'v3', developerKey=api_key)

    # # Define the search query
    # search_query = processed_string + 'mkbhd'

    # # Call the API to search for videos
    # search_response = youtube.search().list(
    #     q=search_query,
    #     part='id',
    #     type='video',
    #     maxResults=1
    # ).execute()

    # # Extract video IDs from the search results
    # video_ids = [item['id']['videoId'] for item in search_response['items']]

    # # Retrieve comments for each video
    # comments_data = []
    # for video_id in video_ids:
    #     next_page_token = None
    #     while True:
    #         response = youtube.commentThreads().list(
    #             part='snippet',
    #             videoId=video_id,
    #             pageToken=next_page_token
    #         ).execute()

    #         # print(response['items'][0]['snippet']['topLevelComment']['snippet'])
            
    #         for item in response['items']:
    #             comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
    #             likes = item['snippet']['topLevelComment']['snippet']['likeCount']
    #             publishedAt = item['snippet']['topLevelComment']['snippet']['publishedAt']
    #             comments_data.append({'Video ID': video_id, 'Likes': likes, 'Time': publishedAt , 'Comment': comment})
            
    #         next_page_token = response.get('nextPageToken')
    #         if not next_page_token:
    #             break


    # df = pd.DataFrame(comments_data)
    df = pd.read_csv("youtube_comments.csv")
    df['Date'] = pd.to_datetime(df['Time'])
    stop_words= ['yourselves', 'between', 'whom', 'itself', 'is', "she's", 'up', 'herself', 'here', 'your', 'each', 
             'we', 'he', 'my', "you've", 'having', 'in', 'both', 'for', 'themselves', 'are', 'them', 'other',
             'and', 'an', 'during', 'their', 'can', 'yourself', 'she', 'until', 'so', 'these', 'ours', 'above', 
             'what', 'while', 'have', 're', 'more', 'only', "needn't", 'when', 'just', 'that', 'were', "don't", 
             'very', 'should', 'any', 'y', 'isn', 'who',  'a', 'they', 'to', 'too', "should've", 'has', 'before',
             'into', 'yours', "it's", 'do', 'against', 'on',  'now', 'her', 've', 'd', 'by', 'am', 'from', 
             'about', 'further', "that'll", "you'd", 'you', 'as', 'how', 'been', 'the', 'or', 'doing', 'such',
             'his', 'himself', 'ourselves',  'was', 'through', 'out', 'below', 'own', 'myself', 'theirs', 
             'me', 'why', 'once',  'him', 'than', 'be', 'most', "you'll", 'same', 'some', 'with', 'few', 'it',
             'at', 'after', 'its', 'which', 'there','our', 'this', 'hers', 'being', 'did', 'of', 'had', 'under',
             'over','again', 'where', 'those', 'then', "you're", 'i', 'because', 'does', 'all', "bot", "yeah", "first"]

    CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});') 
    df["Comment"] = df["Comment"].apply(lambda x: re.sub(CLEANR, '', x))
    df["Comment"] = df["Comment"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    df.head()

    df["Comment"] = df["Comment"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    df["polarity"] = df["Comment"].map(lambda text: TextBlob(text).sentiment.polarity)
    df["sentiment"] = df["polarity"].apply(lambda x: "Negative" if (x <= -0.07) else ("Positive" if (x >= 0.09) else "Neutral"))

    #Filtering data
    review_pos = df[df["sentiment"]=='Positive'].dropna()
    review_neu = df[df["sentiment"]=='Neutral'].dropna()
    review_neg = df[df["sentiment"]=='Negative'].dropna()



    ## Get the bar chart from positive Comment ##
    freq_dict = defaultdict(int)
    for sent in review_pos["Comment"]:
        for word in generate_ngrams(sent):
            freq_dict[word] += 1
    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted.columns = ["word", "wordcount"]
    trace0 = horizontal_bar_chart(fd_sorted.head(25), 'green')

    ## Get the bar chart from neutral Comment ##
    freq_dict = defaultdict(int)
    for sent in review_neu["Comment"]:
        for word in generate_ngrams(sent):
            freq_dict[word] += 1
    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted.columns = ["word", "wordcount"]
    trace1 = horizontal_bar_chart(fd_sorted.head(25), 'grey')

    ## Get the bar chart from negative Comment ##
    freq_dict = defaultdict(int)
    for sent in review_neg["Comment"]:
        for word in generate_ngrams(sent):
            freq_dict[word] += 1
    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted.columns = ["word", "wordcount"]
    trace2 = horizontal_bar_chart(fd_sorted.head(25), 'red')

    # Creating two subplots
    fig = tools.make_subplots(rows=3, cols=1, vertical_spacing=0.04,
                            subplot_titles=["Frequent words of positive Comment", "Frequent words of neutral Comment",
                                            "Frequent words of negative Comment"])
    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 2, 1)
    fig.append_trace(trace2, 3, 1)
    fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(255,255,255)', title="Word Count Distribution")
    img_bytes = io.BytesIO()
    pio.write_image(fig, img_bytes, format='png')
    img_bytes.seek(0)
    
    # Encode image bytes as base64 string
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    json_data = df.to_json(orient='records')
    


    freq_dict = defaultdict(int)
    for sent in review_pos["Comment"]:
        for word in generate_ngrams(sent,2):
            freq_dict[word] += 1
    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted.columns = ["word", "wordcount"]
    trace0 = horizontal_bar_chart(fd_sorted.head(25), 'green')

    ## Get the bar chart from neutral Comment ##
    freq_dict = defaultdict(int)
    for sent in review_neu["Comment"]:
        for word in generate_ngrams(sent,2):
            freq_dict[word] += 1
    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted.columns = ["word", "wordcount"]
    trace1 = horizontal_bar_chart(fd_sorted.head(25), 'grey')

    ## Get the bar chart from negative Comment ##
    freq_dict = defaultdict(int)
    for sent in review_neg["Comment"]:
        for word in generate_ngrams(sent,2):
            freq_dict[word] += 1
    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted.columns = ["word", "wordcount"]
    trace2 = horizontal_bar_chart(fd_sorted.head(25), 'brown')



    # Creating two subplots
    fig = tools.make_subplots(rows=3, cols=1, vertical_spacing=0.04,horizontal_spacing=0.25,
                            subplot_titles=["Bigram plots of Positive Comment", 
                                            "Bigram plots of Neutral Comment",
                                            "Bigram plots of Negative Comment"
                                            ])
    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 2, 1)
    fig.append_trace(trace2, 3, 1)


    fig['layout'].update(height=1000, width=800, paper_bgcolor='rgb(255,255,255)', title="Bigram Distribution")
    img_bytes2 = io.BytesIO()
    pio.write_image(fig, img_bytes2, format='png')
    img_bytes2.seek(0)
    bigram_img_base64 = base64.b64encode(img_bytes2.read()).decode('utf-8')



    for sent in review_pos["Comment"]:
        for word in generate_ngrams(sent,3):
            freq_dict[word] += 1

    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted.columns = ["word", "wordcount"]
    trace0 = horizontal_bar_chart(fd_sorted.head(25), 'green')

    ## Get the bar chart from neutral Comment ##
    freq_dict = defaultdict(int)
    for sent in review_neu["Comment"]:
        for word in generate_ngrams(sent,3):
            freq_dict[word] += 1
    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted.columns = ["word", "wordcount"]
    trace1 = horizontal_bar_chart(fd_sorted.head(25), 'grey')

    ## Get the bar chart from negative Comment ##
    freq_dict = defaultdict(int)
    for sent in review_neg["Comment"]:
        for word in generate_ngrams(sent,3):
            freq_dict[word] += 1
    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted.columns = ["word", "wordcount"]
    trace2 = horizontal_bar_chart(fd_sorted.head(25), 'red')


    # Creating two subplots
    fig = tools.make_subplots(rows=3, cols=1, vertical_spacing=0.04, horizontal_spacing=0.05,
                            subplot_titles=["Tri-gram plots of Positive Comment", 
                                            "Tri-gram plots of Neutral Comment",
                                            "Tri-gram plots of Negative Comment"])
    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 2, 1)
    fig.append_trace(trace2, 3, 1)
    fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(255,255,255)', title="Trigram Distribution")
    img_bytes3 = io.BytesIO()
    pio.write_image(fig, img_bytes3, format='png')
    img_bytes3.seek(0)
    trigram_img_base64 = base64.b64encode(img_bytes3.read()).decode('utf-8')

    # Comment summarization
    all_pos = ""
    all_neg = ""
    all_neu = ""

    for comment in review_pos["Comment"]:
        all_pos += str(comment)+"."
    for comment in review_neu["Comment"]:
        all_neu += str(comment)+"."
    for comment in review_neg["Comment"]:
        all_neg += str(comment)+"."


    response_data = {'input_string': input_string, 'plot_image_monogram': img_base64, 'plot_image_bigram': bigram_img_base64 , 'plot_image_trigram': trigram_img_base64, 
                     'pos_summ': summarize(all_pos, 0.0004)
                     , 'neg_summ': summarize(all_neg, 0.001)
                     , 'neu_summ': summarize(all_neu, 0.0004)}

    # Return the JSON response
    return response_data



@app.route('/searchReddit', methods=['POST'])
def process_string():
    # Get the input string from the request data
    input_string = request.json.get('input_string')

    if not input_string:
        return jsonify({'error': 'Input string is missing'}), 400

    processed_string = process_string(input_string)
    api_key = "AIzaSyAWi-L9FRX0R27eBp3Iy7QfBao_XdijQto"
if __name__ == '__main__':
    app.run(debug=True)
