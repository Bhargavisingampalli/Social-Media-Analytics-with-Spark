import pandas as pd
import pyspark
import re
import os
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, udf, explode, split, size, when
from pyspark.sql.types import StringType, ArrayType, IntegerType, FloatType, StructField, StructType
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import textblob 
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import networkx as nx
from pyvis.network import Network
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from collections import Counter
import dash_bootstrap_components as dbc
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models
import folium
from folium.plugins import HeatMap

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

spark = SparkSession.builder \
    .appName("TwitterAnalysis") \
    .config("spark.hadoop.fs.defaultFS", "file:///") \
    .getOrCreate()


df = spark.read.csv("C:/Users/Bhargavi/Desktop/twitter_data2.csv", header=True, inferSchema=True, multiLine=True)
hashtag_location_map = {

    "#NYC": ("New York City", (40.7128, -74.0060)),
    "#LosAngeles": ("Los Angeles", (34.0522, -118.2437)),
    "#SanFrancisco": ("San Francisco", (37.7749, -122.4194)),
    "#Chicago": ("Chicago", (41.8781, -87.6298)),
    "#Houston": ("Houston", (29.7604, -95.3698)),
    "#Miami": ("Miami", (25.7617, -80.1918)),
    "#Boston": ("Boston", (42.3601, -71.0589)),
    "#Seattle": ("Seattle", (47.6062, -122.3321)),
    "#Dallas": ("Dallas", (32.7767, -96.7970)),
    "#Atlanta": ("Atlanta", (33.7490, -84.3880)),
    # UK
    "#London": ("London", (51.5074, -0.1278)),
    "#Manchester": ("Manchester", (53.4808, -2.2426)),
    "#Birmingham": ("Birmingham", (52.4862, -1.8904)),
    # India
    "#Delhi": ("Delhi", (28.6139, 77.2090)),
    "#Mumbai": ("Mumbai", (19.0760, 72.8777)),
    "#Bombay": ("Mumbai", (19.0760, 72.8777)), # Old name
    "#Hyderabad": ("Hyderabad", (17.3850, 78.4867)),
    "#Bangalore": ("Bangalore", (12.9716, 77.5946)),
    "#Bengaluru": ("Bangalore", (12.9716, 77.5946)), # Synonym
    "#Mangalore": ("Mangalore", (12.9141, 74.8560)),
    "#Chennai": ("Chennai", (13.0827, 80.2707)),
    "#Pune": ("Pune", (18.5204, 73.8567)),
    "#Kolkata": ("Kolkata", (22.5726, 88.3639)),
    "#Ahmedabad": ("Ahmedabad", (23.0225, 72.5714)),
    "#Vizag": ("Visakhapatnam", (17.6868, 83.2185)),
    "#Jaipur": ("Jaipur", (26.9124, 75.7873)),
    "#Lucknow": ("Lucknow", (26.8467, 80.9462)),
    "#Chandigarh": ("Chandigarh", (30.7333, 76.7794)),
    "#Coimbatore": ("Coimbatore", (11.0168, 76.9558)),
    # Europe
    "#Paris": ("Paris", (48.8566, 2.3522)),
    "#Berlin": ("Berlin", (52.5200, 13.4050)),
    "#Rome": ("Rome", (41.9028, 12.4964)),
    "#Madrid": ("Madrid", (40.4168, -3.7038)),
    "#Barcelona": ("Barcelona", (41.3784, 2.1925)),
    "#Zurich": ("Zurich", (47.3769, 8.5417)),
    "#Amsterdam": ("Amsterdam", (52.3676, 4.9041)),
    "#Athens": ("Athens", (37.9838, 23.7275)),
    "#Helsinki": ("Helsinki", (60.1695, 24.9354)),
    # Asia
    "#Tokyo": ("Tokyo", (35.6762, 139.6503)),
    "#Seoul": ("Seoul", (37.5665, 126.9780)),
    "#Singapore": ("Singapore", (1.3521, 103.8198)),
    "#Bangkok": ("Bangkok", (13.7563, 100.5018)),
    "#KualaLumpur": ("Kuala Lumpur", (3.1390, 101.6869)),
    "#Jakarta": ("Jakarta", (-6.2088, 106.8456)),
    "#Dubai": ("Dubai", (25.276987, 55.296249)),
    "#Istanbul": ("Istanbul", (41.0082, 28.9784)),
    # Africa
    "#Cairo": ("Cairo", (30.0444, 31.2357)),
    "#CapeTown": ("Cape Town", (-33.9249, 18.4241)),
    "#Lagos": ("Lagos", (6.5244, 3.3792)),
    # South America
    "#BuenosAires": ("Buenos Aires", (-34.6037, -58.3816)),
    "#Rio": ("Rio de Janeiro", (-22.9068, -43.1729)),
    "#SaoPaulo": ("Sao Paulo", (-23.5505, -46.6333)),
    "#MexicoCity": ("Mexico City", (19.4326, -99.1332)),
    # Canada
    "#Toronto": ("Toronto", (43.6532, -79.3832)),
    "#Vancouver": ("Vancouver", (49.2827, -123.1207)),
    # Russia
    "#Moscow": ("Moscow", (55.7558, 37.6173)),
    # Regional
    "#India": ("India", (20.5937, 78.9629)),
    "#USA": ("USA", (37.0902, -95.7129)),
    "#UK": ("United Kingdom", (55.3781, -3.4360)),
    "#Europe": ("Europe", (54.5260, 15.2551)),
    "#Asia": ("Asia", (34.0479, 100.6197)),
}
# Preprocessing functions
def clean_tweet(text):
    if text is None:
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_hashtags(text):
    if text is None:
        return []
    hashtags = re.findall(r'#(\w+)', text)
    return hashtags

def extract_hashtag_locations(hashtags):
    if not hashtags or not isinstance(hashtags, list):
        return None, None, None
    
    for hashtag in hashtags:
        if hashtag in hashtag_location_map:
            location_name, coords = hashtag_location_map[hashtag]
            return hashtag, location_name, coords
    return None, None, None

# Register the UDF
extract_hashtag_locations_udf = udf(extract_hashtag_locations, 
                                  StructType([
                                      StructField("hashtag", StringType()),
                                      StructField("location_name", StringType()),
                                      StructField("coordinates", ArrayType(FloatType()))
                                  ]))

def extract_mentions(text):
    if text is None:
        return []
    mentions = re.findall(r'@(\w+)', text)
    return mentions

#def extract_location(location):
#    if location is None:
#       return "Unknown"
#    return location

def tokenize_text(text):
    if text is None or text == "":
        return []
    # Tokenize, remove stopwords, and lemmatize
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return tokens

# Register UDFs
clean_tweet_udf = udf(clean_tweet, StringType())
extract_hashtags_udf = udf(extract_hashtags, ArrayType(StringType()))
extract_mentions_udf = udf(extract_mentions, ArrayType(StringType()))
#extract_location_udf = udf(extract_location, StringType())
tokenize_text_udf = udf(tokenize_text, ArrayType(StringType()))

# Apply preprocessing
df = df.withColumn("clean_text", clean_tweet_udf(col("Text")))
df = df.withColumn("hashtags", extract_hashtags_udf(col("Text")))
df = df.withColumn("mentions", extract_mentions_udf(col("Text")))
#df = df.withColumn("location", extract_location_udf(col("Location")))
df = df.withColumn("tokens", tokenize_text_udf(col("clean_text")))

# Sentiment Analysis
def get_sentiment(text):
    if text is None or text == "":
        return 0.0
    try:
        analysis = TextBlob(text)
        return float(analysis.sentiment.polarity)
    except:
        return 0.0

def get_sentiment_label(score):
    if score is None:
        return "Neutral"
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

sentiment_udf = udf(get_sentiment, FloatType())
sentiment_label_udf = udf(get_sentiment_label, StringType())

df = df.withColumn("sentiment_score", sentiment_udf(col("clean_text")))
df = df.withColumn("sentiment", sentiment_label_udf(col("sentiment_score")))
df = df.withColumn("hashtag_location_info", extract_hashtag_locations_udf(col("hashtags")))
df = df.withColumn("location_hashtag", col("hashtag_location_info.hashtag"))
df = df.withColumn("location_name", col("hashtag_location_info.location_name"))
df = df.withColumn("coordinates", col("hashtag_location_info.coordinates"))

# Split coordinates into latitude and longitude
df = df.withColumn("latitude", col("coordinates")[0])
df = df.withColumn("longitude", col("coordinates")[1])

# Convert to Pandas for Dash
df_pd = df.toPandas()

# Time-based features - assuming there's a DateTime column, if not we'll skip these
if "DateTime" in df.columns:
    df = df.withColumn("date", to_date(col("DateTime")))
    df = df.withColumn("hour", hour(col("DateTime")))
    df = df.withColumn("day_of_week", dayofweek(col("DateTime")))

# Convert to Pandas for Dash
df_pd = df.toPandas()

# Fix data types for metrics if they're not already numeric
numeric_cols = ["Likes", "Retweets", "Replies"]
for col in numeric_cols:
    if col in df_pd.columns:
        df_pd[col] = pd.to_numeric(df_pd[col], errors='coerce').fillna(0)

# Generate mock coordinates for geospatial visualization if not available
# This is for demonstration - in a real scenario, you'd use actual coordinates or geocode locations
if "Latitude" not in df_pd.columns or "Longitude" not in df_pd.columns:
    # Generate random coordinates based on usernames instead of locations
    np.random.seed(42)  # For reproducibility
    usernames = df_pd['Username'].unique()
    user_coords = {}
    for user in usernames:
        if pd.notna(user):
            # Generate random coordinates around the world
            user_coords[user] = (
                np.random.uniform(-90, 90),  # Latitude
                np.random.uniform(-180, 180)  # Longitude
            )
    
    # Map coordinates to DataFrame
    df_pd["Latitude"] = df_pd['Username'].map(lambda x: user_coords.get(x, (np.nan, np.nan))[0] if pd.notna(x) else np.nan)
    df_pd["Longitude"] = df_pd['Username'].map(lambda x: user_coords.get(x, (np.nan, np.nan))[1] if pd.notna(x) else np.nan )

# Update the location-based charts to use usernames instead:

# Generate Word Cloud
def generate_wordcloud(texts, mask=None):
    # Check if input is empty or all values are NA
    if texts.empty or texts.isna().all():
        # Create a simple wordcloud with a placeholder message
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate("No data available")
    else:
        # Filter out NA values and join the texts
        text_data = " ".join([str(text) for text in texts if pd.notna(text)])
        if not text_data.strip():  # Check if we have any non-whitespace text
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate("No valid text data")
        else:
            wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                mask=mask, collocations=False).generate(text_data)
    
    # Convert to image
    img = BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

# Topic Modeling function
def perform_topic_modeling(documents, num_topics=5):
    if not documents or all(not doc for doc in documents):
        return [{"topic": i, "words": ["No data available"]} for i in range(num_topics)]
    
    # Create a dictionary and corpus
    dictionary = corpora.Dictionary([doc.split() for doc in documents if doc])
    corpus = [dictionary.doc2bow(doc.split()) for doc in documents if doc]
    
    # Train LDA model
    lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15)
    
    # Extract topics
    topics = []
    for i in range(num_topics):
        topic_words = dict(lda_model.show_topic(i, topn=10))
        topics.append({
            "topic": i,
            "words": list(topic_words.keys())
        })
    
    return topics

# Network Analysis
def build_network_graph(df, filter_by=None):
    G = nx.Graph()
    
    # Function to add nodes and edges with attributes
    def add_interaction(source, target, type_interaction, tweet_id=None):
        if source and target and source != target:
            # Add nodes with attributes if they don't exist
            if not G.has_node(source):
                G.add_node(source, type='user', size=5)
            if not G.has_node(target):
                G.add_node(target, type='user', size=5)
            
            # Add or update edge
            if G.has_edge(source, target):
                G[source][target]['weight'] += 1
                G[source][target]['types'].add(type_interaction)
            else:
                G.add_edge(source, target, weight=1, types={type_interaction}, tweets=set())
            
            if tweet_id:
                G[source][target]['tweets'].add(tweet_id)
    
    # Process dataframe to extract network relationships
    for _, row in df.iterrows():
        source = row.get('Username')
        
        if pd.isna(source) or not source:
            continue
        
        # Filter by a specific attribute if requested
        if filter_by and filter_by != 'all':
            if filter_by == 'positive' and row['sentiment'] != 'Positive':
                continue
            elif filter_by == 'negative' and row['sentiment'] != 'Negative':
                continue
        
        # Add mention relationships
        mentions = row.get('mentions', [])
        if mentions and isinstance(mentions, list):
            for mention in mentions:
                add_interaction(source, mention, 'mention', row.get('id'))
        
        # Add retweet relationships
        text = row.get('Text', '')
        if isinstance(text, str) and text.startswith('RT @'):
            rt_user = re.search(r'RT @(\w+)', text)
            if rt_user:
                retweeted_user = rt_user.group(1)
                add_interaction(source, retweeted_user, 'retweet', row.get('id'))
    
    # Simplify graph if too large
    if len(G.nodes()) > 100:
        # Keep only nodes with significant connections
        to_remove = [node for node, degree in dict(G.degree()).items() if degree < 2]
        G.remove_nodes_from(to_remove)
    
    # Set node sizes based on degree centrality
    if G.number_of_nodes() > 0:
        centrality = nx.degree_centrality(G)
        for node in G.nodes():
            G.nodes[node]['size'] = 5 + 20 * centrality.get(node, 0)
    
    # Generate HTML visualization
    if G.number_of_edges() > 0:
        net = Network(height="600px", width="100%", notebook=True, bgcolor="#222222", font_color="white")
        
        # Add nodes and edges with attributes
        for node, attr in G.nodes(data=True):
            net.add_node(
                node, 
                size=attr.get('size', 5),
                title=f"User: {node}",
                color="#00a0dc" if attr.get('type') == 'user' else "#d62976"
            )
        
        for source, target, attr in G.edges(data=True):
            interaction_types = ", ".join(attr.get('types', ['interaction']))
            net.add_edge(
                source, 
                target, 
                value=attr.get('weight', 1),
                title=f"{interaction_types} ({attr.get('weight', 1)} times)"
            )
        
        net.save_graph("network.html")
        with open("network.html", "r") as f:
            html_content = f.read()
        return html_content
    else:
        # Return empty graph message if no relationships found
        return "<div style='text-align:center; padding:50px;'><h3>No network relationships found with current filters.</h3></div>"

# Create a location heat map
def create_heatmap(df):
    # Filter for rows with valid coordinates
    geo_df = df[df['latitude'].notna() & df['longitude'].notna()]
    
    if len(geo_df) == 0:
        # Return a default map if no coordinates
        m = folium.Map(location=[20, 0], zoom_start=2)
        folium.TileLayer('cartodbdark_matter').add_to(m)
        return m._repr_html_()
    
    # Create map centered on the mean of all coordinates
    m = folium.Map(
        location=[geo_df['latitude'].mean(), geo_df['longitude'].mean()],
        zoom_start=2,
        tiles='cartodbdark_matter'
    )
    
    # Add heat map layer
    heat_data = [[row['latitude'], row['longitude']] for _, row in geo_df.iterrows()]
    HeatMap(heat_data).add_to(m)
    
    # Add markers for top locations
    top_locations = geo_df['location_name'].value_counts().head(10).index
    for loc in top_locations:
        loc_df = geo_df[geo_df['location_name'] == loc]
        popup_text = f"""
        <b>{loc}</b><br>
        Tweets: {len(loc_df)}<br>
        Top Hashtag: {loc_df['location_hashtag'].mode()[0]}
        """
        folium.Marker(
            location=[loc_df['latitude'].iloc[0], loc_df['longitude'].iloc[0]],
            popup=popup_text,
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
    
    # Save and return HTML
    m.save('heatmap.html')
    with open('heatmap.html', 'r') as f:
        return f.read()

# Initialize the Dash app with Bootstrap for styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Navbar with logo

# Tab navigation for different sections
tab_navigation = dbc.Tabs(
    [
        dbc.Tab(label="Overview", tab_id="overview"),
        dbc.Tab(label="Trend Analysis", tab_id="trends"),
        dbc.Tab(label="User Engagement", tab_id="engagement"),
        dbc.Tab(label="Geospatial Analysis", tab_id="geo"),
        dbc.Tab(label="Network Analysis", tab_id="network"),
        dbc.Tab(label="Topic Analysis", tab_id="topics"),
    ],
    id="tabs",
    active_tab="overview",
    className="mb-3"
)

# Card components for the dashboard
key_metrics_card = dbc.Card(
    [
        dbc.CardHeader("Key Metrics"),
        dbc.CardBody(
            [
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            html.H2(id="total-tweets", children="0"),
                            html.P("Total Tweets")
                        ], className="text-center p-3 bg-primary text-white")
                    ),
                    dbc.Col(
                        dbc.Card([
                            html.H2(id="total-users", children="0"),
                            html.P("Unique Users")
                        ], className="text-center p-3 bg-info text-white")
                    ),
                    dbc.Col(
                        dbc.Card([
                            html.H2(id="total-engagement", children="0"),
                            html.P("Total Engagement")
                        ], className="text-center p-3 bg-success text-white")
                    ),
                    dbc.Col(
                        dbc.Card([
                            html.H2(id="sentiment-ratio", children="0%"),
                            html.P("Positive Sentiment")
                        ], className="text-center p-3 bg-warning text-dark")
                    ),
                ])
            ]
        )
    ]
)

# Overview Tab Content
overview_content = dbc.Container([
    key_metrics_card,
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sentiment Distribution"),
                dbc.CardBody(dcc.Graph(id="sentiment-pie"))
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Trending Topics"),
                dbc.CardBody(html.Img(id="wordcloud-overview", style={"width": "100%"}))
            ])
        ], width=6)
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Top Hashtags"),
                dbc.CardBody(dcc.Graph(id="top-hashtags"))
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Engagement Overview"),
                dbc.CardBody(dcc.Graph(id="engagement-metrics"))
            ])
        ], width=6)
    ])
])

# Trend Analysis Tab Content
trends_content = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Trending Hashtags Over Time"),
                dbc.CardBody(dcc.Graph(id="hashtag-trends"))
            ])
        ], width=12)
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Word Cloud Analysis"),
                dbc.CardBody(html.Img(id="detailed-wordcloud", style={"width": "100%"}))
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Top Emerging Topics"),
                dbc.CardBody(dcc.Graph(id="emerging-topics"))
            ])
        ], width=6)
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sentiment Trends"),
                dbc.CardBody(dcc.Graph(id="sentiment-trends"))
            ])
        ], width=12)
    ])
])

# User Engagement Tab Content
engagement_content = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("User Engagement Metrics"),
                dbc.CardBody(dcc.Graph(id="user-engagement-metrics"))
            ])
        ], width=12)
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Top Users by Engagement"),
                dbc.CardBody(dcc.Graph(id="top-users"))
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Engagement by Content Type"),
                dbc.CardBody(dcc.Graph(id="content-engagement"))
            ])
        ], width=6)
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("User Growth Over Time"),
                dbc.CardBody(dcc.Graph(id="user-growth"))
            ])
        ], width=12)
    ])
])

# Geospatial Analysis Tab Content
geo_content = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Global Tweet Distribution"),
                dbc.CardBody([
                    html.Iframe(
                        id="geo-heatmap",
                        srcDoc=create_heatmap(df_pd),
                        style={"width": "100%", "height": "500px", "border": "none"}
                    )
                ])
            ])
        ], width=12)
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Top Locations by Tweet Volume"),
                dbc.CardBody(dcc.Graph(id="location-barchart"))
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sentiment by Location"),
                dbc.CardBody(dcc.Graph(id="geo-sentiment"))
            ])
        ], width=6)
    ])
])

# Network Analysis Tab Content
network_content = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    "User Connection Network",
                    dbc.ButtonGroup([
                        dbc.Button("All", color="primary", id="network-all", className="me-1", n_clicks=0),
                        dbc.Button("Positive", outline=True, color="success", id="network-positive", className="me-1", n_clicks=0),
                        dbc.Button("Negative", outline=True, color="danger", id="network-negative", n_clicks=0)
                    ], className="float-end")
                ]),
                dbc.CardBody([
                    html.Iframe(
                        id="network-graph",
                        srcDoc=build_network_graph(df_pd),
                        style={"width": "100%", "height": "600px", "border": "none"}
                    )
                ])
            ])
        ], width=12)
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Top Influencers"),
                dbc.CardBody(dcc.Graph(id="influencer-chart"))
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Community Analysis"),
                dbc.CardBody(dcc.Graph(id="community-chart"))
            ])
        ], width=6)
    ])
])

# Topic Analysis Tab Content
topics_content = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Topic Distribution"),
                dbc.CardBody(dcc.Graph(id="topic-distribution"))
            ])
        ], width=12)
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Topic Keywords"),
                dbc.CardBody(id="topic-keywords")
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Topic Evolution"),
                dbc.CardBody(dcc.Graph(id="topic-evolution"))
            ])
        ], width=6)
    ])
])

# Main app layout
app.layout = html.Div([
    
    dbc.Container([
        html.H1("Twitter Social Media Analytics Dashboard", className="text-center mb-4 p-4"),
        tab_navigation,
        html.Div(id="tab-content")
    ], fluid=True)
])

# Callbacks

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab_content(active_tab):
    if active_tab == "overview":
        return overview_content
    elif active_tab == "trends":
        return trends_content
    elif active_tab == "engagement":
        return engagement_content
    elif active_tab == "geo":
        return geo_content
    elif active_tab == "network":
        return network_content
    elif active_tab == "topics":
        return topics_content
    return "Select a tab"

# Overview tab callbacks
@app.callback(
    [
        Output("total-tweets", "children"),
        Output("total-users", "children"),
        Output("total-engagement", "children"),
        Output("sentiment-ratio", "children")
    ],
    Input("tabs", "active_tab")
)
def update_key_metrics(tab):
    total_tweets = len(df_pd)
    unique_users = df_pd['Username'].nunique()
    
    # Calculate total engagement
    engagement_cols = ['Likes', 'Retweets', 'Replies']
    total_engagement = sum(df_pd[col].sum() for col in engagement_cols if col in df_pd.columns)
    
    # Calculate positive sentiment ratio
    pos_count = len(df_pd[df_pd['sentiment'] == 'Positive'])
    pos_ratio = f"{int(pos_count / total_tweets * 100)}%" if total_tweets > 0 else "0%"
    
    return f"{total_tweets:,}", f"{unique_users:,}", f"{int(total_engagement):,}", pos_ratio

@app.callback(
    Output("sentiment-pie", "figure"),
    Input("tabs", "active_tab")
)
def update_sentiment_pie(tab):
    sentiment_counts = df_pd['sentiment'].value_counts()
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Tweet Sentiment Distribution",
        color_discrete_map={
            'Positive': '#28a745',
            'Neutral': '#6c757d',
            'Negative': '#dc3545'
        }
    )
    fig.update_layout(legend_title_text='Sentiment', template="plotly_dark")
    return fig


@app.callback(
    Output("wordcloud-overview", "src"),
    Input("tabs", "active_tab")
)
def update_wordcloud_overview(tab):
    return generate_wordcloud(df_pd["clean_text"].dropna())

@app.callback(
    Output("top-hashtags", "figure"),
    Input("tabs", "active_tab")
)
def update_top_hashtags(tab):
    # Extract all hashtags from all tweets
    all_hashtags = []
    for hashtags in df_pd['hashtags'].dropna():
        if isinstance(hashtags, list):
            all_hashtags.extend(hashtags)
    
    # Count frequencies
    hashtag_counts = Counter(all_hashtags).most_common(10)
    
    if not hashtag_counts:
        # Return empty chart with message if no hashtags
        fig = go.Figure()
        fig.add_annotation(
            text="No hashtags found in the dataset",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Create horizontal bar chart
    hashtags = [h[0] for h in hashtag_counts]
    counts = [h[1] for h in hashtag_counts]
    
    fig = px.bar(
        x=counts,
        y=hashtags,
        orientation='h',
        text=counts,
        title="Top 10 Hashtags",
        labels={'x': 'Count', 'y': 'Hashtag'}
    )
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, template="plotly_dark")
    fig.update_traces(marker_color='#1DA1F2')  # Twitter blue
    
    return fig

@app.callback(
    Output("engagement-metrics", "figure"),
    Input("tabs", "active_tab")
)
def update_engagement_metrics(tab):
    try:
        # Get engagement metrics
        metrics = ['Likes', 'Retweets', 'Replies']
        engagement = []
        
        for metric in metrics:
            if metric in df_pd.columns:
                engagement.append({'Metric': metric, 'Count': int(df_pd[metric].sum())})
        
        if not engagement:
            # Return empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="No engagement metrics found in the dataset",
                showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        # Create bar chart
        engagement_df = pd.DataFrame(engagement)
        
        fig = px.bar(
            engagement_df,
            x='Metric',
            y='Count',
            text='Count',
            color='Metric',
            title="Engagement Overview",
            color_discrete_map={
                'Likes': '#4CAF50',
                'Retweets': '#2196F3',
                'Replies': '#9C27B0'
            }
        )
        
        fig.update_layout(template="plotly_dark")
        
        return fig
    except Exception as e:
        print(f"Error in update_engagement_metrics: {e}")
        # Return empty chart with error message
        fig = go.Figure()
        fig.add_annotation(
            text="Error generating engagement metrics",
            showarrow=False,
            font=dict(size=20)
        )
        return fig

# Trend Analysis tab callbacks
@app.callback(
    Output("detailed-wordcloud", "src"),
    Input("tabs", "active_tab")
)
def update_detailed_wordcloud(tab):
    return generate_wordcloud(df_pd["clean_text"].dropna())

@app.callback(
    Output("hashtag-trends", "figure"),
    Input("tabs", "active_tab")
)
def update_hashtag_trends(tab):
    # For demonstration, we'll create a simulated time series
    # In a real app, you'd use the actual timestamps from your data
    
    # Get top 5 hashtags
    all_hashtags = []
    for hashtags in df_pd['hashtags'].dropna():
        if isinstance(hashtags, list):
            all_hashtags.extend(hashtags)
    
    top_hashtags = [h for h, _ in Counter(all_hashtags).most_common(5)]
    
    if not top_hashtags:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No hashtags found for trend analysis",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Create simulated time series data
    np.random.seed(42)
    days = 10
    start_date = datetime.strptime("13/04/2025", "%d/%m/%Y")
    dates = [(start_date - timedelta(days=days - i)).strftime('%Y-%m-%d') for i in range(days)]
    
    # Generate trend data
    # Generate trend data
    trend_data = []
    
    for hashtag in top_hashtags:
        # Create a simulated trend line for each hashtag
        counts = np.random.randint(5, 100, size=days) * (1 + 0.5 * np.sin(np.linspace(0, np.pi, days)))
        for i, date in enumerate(dates):
            trend_data.append({
                'date': date,
                'hashtag': hashtag,
                'count': int(counts[i])
            })
    
    trend_df = pd.DataFrame(trend_data)
    
    # Create line chart
    fig = px.line(
        trend_df,
        x='date',
        y='count',
        color='hashtag',
        title="Hashtag Trend Analysis",
        labels={'date': 'Date', 'count': 'Mentions', 'hashtag': 'Hashtag'}
    )
    
    fig.update_layout(template="plotly_dark")
    
    return fig

@app.callback(
    Output("emerging-topics", "figure"),
    Input("tabs", "active_tab")
)
def update_emerging_topics(tab):
    # Extract topics from cleaned text using CountVectorizer for demonstration
    # In a real app, you might use more sophisticated NLP techniques
    
    cleaned_texts = [text for text in df_pd["clean_text"].dropna() if text.strip()]
    
    if not cleaned_texts:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No text data available for topic extraction",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Extract top terms
    try:
        vectorizer = CountVectorizer(max_features=50, min_df=2, stop_words='english')
        X = vectorizer.fit_transform(cleaned_texts)
        terms = vectorizer.get_feature_names_out()
        term_counts = np.sum(X.toarray(), axis=0)
        
        # Create dataframe with terms and counts
        term_df = pd.DataFrame({'term': terms, 'count': term_counts})
        term_df = term_df.sort_values('count', ascending=False).head(15)
        
        # Create horizontal bar chart
        fig = px.bar(
            term_df,
            x='count',
            y='term',
            orientation='h',
            title="Emerging Topics",
            labels={'count': 'Mentions', 'term': 'Topic'}
        )
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, template="plotly_dark")
        fig.update_traces(marker_color='#FF9800')  # Orange
        
        return fig
    except Exception as e:
        print(f"Error in update_emerging_topics: {e}")
        # Return empty chart with error message
        fig = go.Figure()
        fig.add_annotation(
            text="Error extracting topics",
            showarrow=False,
            font=dict(size=20)
        )
        return fig

@app.callback(
    Output("sentiment-trends", "figure"),
    Input("tabs", "active_tab")
)
def update_sentiment_trends(tab):
    # For demonstration, we'll create a simulated time series for sentiment trends
    # In a real app, you'd use the actual timestamps from your data
    
    # Create simulated time series data
    np.random.seed(41)
    days = 10
    start_date = datetime.strptime("13/04/2025", "%d/%m/%Y")
    dates = [(start_date - timedelta(days=days - i)).strftime('%Y-%m-%d') for i in range(days)]
    
    # Generate sentiment trend data
    sentiments = ['Positive', 'Neutral', 'Negative']
    trend_data = []
    
    # Base values for each sentiment
    pos_base = 50
    neu_base = 30
    neg_base = 20
    
    for i, date in enumerate(dates):
        # Create fluctuating values with a trend
        pos_val = pos_base + np.random.randint(-5, 10)
        neu_val = neu_base + np.random.randint(-3, 5)
        neg_val = neg_base + np.random.randint(-2, 5)
        
        # Ensure values stay positive
        pos_val = max(pos_val, 5)
        neu_val = max(neu_val, 5)
        neg_val = max(neg_val, 5)
        
        # Normalize to sum to 100
        total = pos_val + neu_val + neg_val
        pos_pct = pos_val / total * 100
        neu_pct = neu_val / total * 100
        neg_pct = neg_val / total * 100
        
        trend_data.append({'date': date, 'sentiment': 'Positive', 'percentage': pos_pct})
        trend_data.append({'date': date, 'sentiment': 'Neutral', 'percentage': neu_pct})
        trend_data.append({'date': date, 'sentiment': 'Negative', 'percentage': neg_pct})
    
    sent_trend_df = pd.DataFrame(trend_data)
    
    # Create line chart
    fig = px.line(
        sent_trend_df,
        x='date',
        y='percentage',
        color='sentiment',
        title="Sentiment Trends Over Time",
        labels={'date': 'Date', 'percentage': 'Percentage', 'sentiment': 'Sentiment'},
        color_discrete_map={
            'Positive': '#28a745',
            'Neutral': '#6c757d',
            'Negative': '#dc3545'
        }
    )
    
    fig.update_layout(template="plotly_dark", yaxis_title="Percentage (%)")
    
    return fig

# User Engagement tab callbacks
@app.callback(
    Output("user-engagement-metrics", "figure"),
    Input("tabs", "active_tab")
)
def update_user_engagement_metrics(tab):
    # Calculate engagement metrics per user
    engagement_cols = ['Likes', 'Retweets', 'Replies']
    
    # Check which columns are available
    available_cols = [col for col in engagement_cols if col in df_pd.columns]
    
    if not available_cols:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No engagement metrics found in the dataset",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Aggregate engagement metrics by user
    user_engagement = df_pd.groupby('Username')[available_cols].sum().reset_index()
    
    # Calculate total engagement
    user_engagement['Total Engagement'] = user_engagement[available_cols].sum(axis=1)
    
    # Sort by total engagement and get top 15
    user_engagement = user_engagement.sort_values('Total Engagement', ascending=False).head(15)
    user_engagement = user_engagement[user_engagement['Username'].astype(str).str.strip() != '0']

    
    # Create stacked bar chart
    fig = go.Figure()
    
    for col in available_cols:
        fig.add_trace(
            go.Bar(
                x=user_engagement['Username'],
                y=user_engagement[col],
                name=col
            )
        )
    
    fig.update_layout(
        title="Top Users by Engagement",
        xaxis_title="Username",
        yaxis_title="Engagement Count",
        barmode='stack',
        template="plotly_dark"
    )
    
    return fig

@app.callback(
    Output("top-users", "figure"),
    Input("tabs", "active_tab")
)
def update_top_users(tab):
    # Get top users by tweet count
    user_tweet_counts = df_pd['Username'].value_counts().reset_index()
    user_tweet_counts.columns = ['Username', 'Tweet Count']
    top_users = user_tweet_counts.head(10)
    
    if len(top_users) == 0:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No user data available",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Create horizontal bar chart
    fig = px.bar(
        top_users,
        x='Tweet Count',
        y='Username',
        orientation='h',
        text='Tweet Count',
        title="Most Active Users",
        labels={'Tweet Count': 'Number of Tweets', 'Username': 'User'}
    )
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, template="plotly_dark")
    fig.update_traces(marker_color='#00bcd4')  # Cyan
    
    return fig

@app.callback(
    Output("content-engagement", "figure"),
    Input("tabs", "active_tab")
)
def update_content_engagement(tab):
    # For demonstration, we'll categorize content and show engagement
    # In a real app, you'd use actual content categories from your data
    
    # Define content types (e.g., based on presence of media, links, etc.)
    def categorize_content(row):
        text = row.get('Text', '')
        if not isinstance(text, str):
            return 'Other'
        
        if 'http' in text or 'www' in text:
            return 'Links'
        elif '#' in text:
            return 'Hashtags'
        elif '@' in text:
            return 'Mentions'
        elif len(text) > 140:
            return 'Long Text'
        else:
            return 'Short Text'
    
    # Add content type column
    df_pd['ContentType'] = df_pd.apply(categorize_content, axis=1)
    
    # Check which engagement columns are available
    engagement_cols = ['Likes', 'Retweets', 'Replies']
    available_cols = [col for col in engagement_cols if col in df_pd.columns]
    
    if not available_cols:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No engagement metrics found in the dataset",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Calculate average engagement by content type
    content_engagement = df_pd.groupby('ContentType')[available_cols].mean().reset_index()
    
    # Create grouped bar chart
    fig = px.bar(
        content_engagement,
        x='ContentType',
        y=available_cols,
        title="Engagement by Content Type",
        barmode='group',
        labels={'value': 'Average Engagement', 'variable': 'Metric', 'ContentType': 'Content Type'}
    )
    
    fig.update_layout(template="plotly_dark")
    
    return fig

@app.callback(
    Output("user-growth", "figure"),
    Input("tabs", "active_tab")
)
def update_user_growth(tab):
    # For demonstration, we'll create a simulated time series for user growth
    # In a real app, you'd use the actual timestamps from your data
    
    # Create simulated time series data
    np.random.seed(43)
    days = 30
    start_date = datetime.strptime("13/04/2025", "%d/%m/%Y")
    dates = [(start_date - timedelta(days=days - i)).strftime('%Y-%m-%d') for i in range(days)]
    
    # Generate growth data with an increasing trend
    base = 1000
    growth_data = []
    
    for i, date in enumerate(dates):
        # Create an increasing trend with some random fluctuation
        new_users = int(base * (1 + 0.03 * i) + np.random.randint(-20, 50))
        cumulative = base + int(i * base * 0.05) + np.random.randint(0, 100)
        
        growth_data.append({
            'date': date,
            'metric': 'New Users',
            'value': new_users
        })
        
        growth_data.append({
            'date': date,
            'metric': 'Cumulative Users',
            'value': cumulative
        })
    
    growth_df = pd.DataFrame(growth_data)
    
    # Create line chart
    fig = px.line(
        growth_df,
        x='date',
        y='value',
        color='metric',
        title="User Growth Over Time",
        labels={'date': 'Date', 'value': 'User Count', 'metric': 'Metric'}
    )
    
    fig.update_layout(template="plotly_dark")
    
    return fig

# Geospatial Analysis tab callbacks


@app.callback(
    Output("location-barchart", "figure"),
    Input("tabs", "active_tab")
)
def update_location_chart(tab):
    if 'location_name' not in df_pd.columns:
        fig = go.Figure()
        fig.add_annotation(text="No location data available", showarrow=False)
        return fig
    
    # Count tweets by location
    location_counts = df_pd['location_name'].value_counts().reset_index()
    location_counts.columns = ['Location', 'Tweet Count']
    top_locations = location_counts.head(10)
    
    if len(top_locations) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No location data available", showarrow=False)
        return fig
    
    # Create horizontal bar chart
    fig = px.bar(
        top_locations,
        x='Tweet Count',
        y='Location',
        orientation='h',
        title="Top Locations by Hashtag Mentions",
        labels={'Tweet Count': 'Number of Tweets', 'Location': 'Location'}
    )
    fig.update_layout(template="plotly_dark")
    return fig

# Update the geo-sentiment callback
@app.callback(
    Output("geo-sentiment", "figure"),
    Input("tabs", "active_tab")
)
def update_geo_sentiment(tab):
    if 'location_name' not in df_pd.columns or 'sentiment' not in df_pd.columns:
        fig = go.Figure()
        fig.add_annotation(text="Location or sentiment data not available", showarrow=False)
        return fig
    
    # Get sentiment counts by location
    location_sentiment = df_pd.groupby(['location_name', 'sentiment']).size().reset_index(name='count')
    
    # Calculate percentages
    total_by_location = location_sentiment.groupby('location_name')['count'].transform('sum')
    location_sentiment['percentage'] = location_sentiment['count'] / total_by_location * 100
    
    # Get top 10 locations by tweet volume
    top_locations = df_pd['location_name'].value_counts().nlargest(10).index.tolist()
    location_sentiment = location_sentiment[location_sentiment['location_name'].isin(top_locations)]
    
    # Create stacked bar chart
    fig = px.bar(
        location_sentiment,
        x='location_name',
        y='percentage',
        color='sentiment',
        title="Sentiment Distribution by Location",
        labels={'location_name': 'Location', 'percentage': 'Percentage'},
        color_discrete_map={
            'Positive': '#28a745',
            'Neutral': '#6c757d',
            'Negative': '#dc3545'
        }
    )
    fig.update_layout(template="plotly_dark")
    return fig

# Network Analysis tab callbacks
@app.callback(
    Output("network-graph", "srcDoc"),
    [
        Input("network-all", "n_clicks"),
        Input("network-positive", "n_clicks"),
        Input("network-negative", "n_clicks")
    ]
)
def update_network_graph(all_clicks, positive_clicks, negative_clicks):
    ctx = callback_context
    if not ctx.triggered:
        # Default to all
        return build_network_graph(df_pd)
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == "network-positive":
            return build_network_graph(df_pd, filter_by="positive")
        elif button_id == "network-negative":
            return build_network_graph(df_pd, filter_by="negative")
        else:  # Default or "network-all"
            return build_network_graph(df_pd)

@app.callback(
    Output("influencer-chart", "figure"),
    Input("tabs", "active_tab")
)
def update_influencer_chart(tab):
    # For demonstration, we'll calculate influence based on mentions and engagement
    # In a real app, you'd use network centrality or other social network metrics
    
    # Extract mentions data
    all_mentions = []
    for mentions in df_pd['mentions'].dropna():
        if isinstance(mentions, list):
            all_mentions.extend(mentions)
    
    # Count mentions
    mention_counts = Counter(all_mentions).most_common(10)
    
    if not mention_counts:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No mention data available for influencer analysis",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Create dataframe
    influencer_df = pd.DataFrame(mention_counts, columns=['Username', 'Mentions'])
    
    # Create horizontal bar chart
    fig = px.bar(
        influencer_df,
        x='Mentions',
        y='Username',
        orientation='h',
        text='Mentions',
        title="Top Influencers by Mentions",
        labels={'Mentions': 'Number of Mentions', 'Username': 'User'}
    )
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, template="plotly_dark")
    fig.update_traces(marker_color='#e91e63')  # Pink
    
    return fig

@app.callback(
    Output("community-chart", "figure"),
    Input("tabs", "active_tab")
)
def update_community_chart(tab):
    # For demonstration, we'll simulate community data
    # In a real app, you'd use actual community detection algorithms on the network
    
    # Create simulated community data
    np.random.seed(44)
    communities = ['Tech', 'Politics', 'Entertainment', 'Sports', 'Business']
    community_sizes = np.random.randint(20, 100, size=len(communities))
    community_engagement = np.random.randint(50, 500, size=len(communities))
    
    community_df = pd.DataFrame({
        'Community': communities,
        'Size': community_sizes,
        'Engagement': community_engagement
    })
    
    # Create bubble chart
    fig = px.scatter(
        community_df,
        x='Size',
        y='Engagement',
        size='Size',
        text='Community',
        title="Community Analysis",
        labels={'Size': 'Community Size', 'Engagement': 'Average Engagement'},
        color='Community'
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(template="plotly_dark")
    
    return fig

# Topic Analysis tab callbacks
@app.callback(
    Output("topic-distribution", "figure"),
    Input("tabs", "active_tab")
)
def update_topic_distribution(tab):
    # Run topic modeling
    cleaned_texts = [text for text in df_pd["clean_text"].dropna() if text.strip()]
    
    if not cleaned_texts:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No text data available for topic modeling",
            showarrow=False,
            font=dict(size=20)
        )
        return fig

    # For demonstration, we'll create simulated topic distributions
    # In a real app, you'd use the actual output from your topic model
    
    np.random.seed(45)
    topics = ['Politics', 'Technology', 'Entertainment', 'Sports', 'Business', 'Health']
    topic_counts = np.random.randint(50, 300, size=len(topics))
    
    topic_df = pd.DataFrame({
        'Topic': topics,
        'Count': topic_counts
    })
    
    # Create pie chart
    fig = px.pie(
        topic_df,
        values='Count',
        names='Topic',
        title="Topic Distribution"
    )
    
    fig.update_layout(template="plotly_dark")
    
    return fig

@app.callback(
    Output("topic-keywords", "children"),
    Input("tabs", "active_tab")
)
def update_topic_keywords(tab):
    # For demonstration, we'll use simulated topic keywords
    # In a real app, you'd extract these from your topic model
    
    topics = {
        'Politics': ['government', 'president', 'election', 'policy', 'vote', 'democracy', 'congress'],
        'Technology': ['innovation', 'digital', 'software', 'tech', 'ai', 'device', 'app'],
        'Entertainment': ['movie', 'music', 'celebrity', 'film', 'actor', 'award', 'star'],
        'Sports': ['game', 'team', 'player', 'win', 'championship', 'score', 'season'],
        'Business': ['company', 'market', 'invest', 'economy', 'stock', 'growth', 'revenue'],
        'Health': ['medical', 'health', 'doctor', 'patient', 'treatment', 'hospital', 'care']
    }
    
    # Create cards for each topic
    topic_cards = []
    
    for topic, keywords in topics.items():
        card = dbc.Card([
            dbc.CardHeader(topic),
            dbc.CardBody([
                html.Div([
                    dbc.Badge(keyword, color="info", className="me-1 mb-1")
                    for keyword in keywords
                ])
            ])
        ], className="mb-3")
        
        topic_cards.append(card)
    
    # Arrange in rows of 3
    rows = []
    for i in range(0, len(topic_cards), 3):
        rows.append(
            dbc.Row([
                dbc.Col(card, width=4)
                for card in topic_cards[i:i+3]
            ])
        )
    
    return html.Div(rows)

@app.callback(
    Output("topic-evolution", "figure"),
    Input("tabs", "active_tab")
)
def update_topic_evolution(tab):
    # For demonstration, we'll create simulated time series for topic evolution
    # In a real app, you'd use the actual timestamps and topic distributions from your data
    
    # Create simulated time series data
    np.random.seed(46)
    days = 10
    start_date = datetime.strptime("13/04/2025", "%d/%m/%Y")
    dates = [(start_date - timedelta(days=days - i)).strftime('%Y-%m-%d') for i in range(days)]
    
    # Generate topic evolution data
    topics = ['Politics', 'Technology', 'Entertainment']
    evolution_data = []
    
    for topic in topics:
        # Create a trend line for each topic
        if topic == 'Politics':
            # Decreasing trend
            values = np.linspace(60, 30, days) + np.random.randint(-5, 5, size=days)
        elif topic == 'Technology':
            # Increasing trend
            values = np.linspace(20, 50, days) + np.random.randint(-5, 5, size=days)
        else:  # Entertainment
            # Stable with fluctuations
            values = np.linspace(30, 35, days) + np.random.randint(-7, 7, size=days)
        
        for i, date in enumerate(dates):
            evolution_data.append({
                'date': date,
                'topic': topic,
                'percentage': values[i]
            })
    
    evolution_df = pd.DataFrame(evolution_data)
    
    # Create line chart
    fig = px.line(
        evolution_df,
        x='date',
        y='percentage',
        color='topic',
        title="Topic Evolution Over Time",
        labels={'date': 'Date', 'percentage': 'Percentage', 'topic': 'Topic'}
    )
    
    fig.update_layout(template="plotly_dark", yaxis_title="Percentage (%)")
    
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)