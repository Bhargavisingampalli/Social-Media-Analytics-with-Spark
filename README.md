# Social-Media-Analytics-with-Spark

Social Media Analytics Platform is a scalable data analysis solution built using Apache Spark for big data processing and Streamlit for interactive visualizations. The project focuses on extracting insights from Twitter (X) data — including trending topics, user engagement metrics, geospatial activity, and network relationships — to understand social media dynamics at scale.

## Features

- **Trending Topics and Hashtags**: Identifies frequently discussed hashtags and keywords.
- **User Engagement Analytics**: Measures likes, retweets, mentions, and replies to determine engagement.
- **Geospatial Analysis**: Maps tweet distributions to understand regional activity.
- **Network Graphs**: Builds user interaction networks to identify key influencers and communities.

## Tech Stack

- Apache Spark (Batch + Streaming)
- Python (Pandas, NLTK, Matplotlib)
- Streamlit (for visualization)
- Twitter API (for data collection)
- Python Socket Server (to simulate real-time streaming)

## Project Workflow

1. **Data Collection**:
   - Twitter API was used to collect tweet data and store it in a CSV file.

2. **Preprocessing**:
   - The CSV data was cleaned, and key features were extracted (hashtags, mentions, timestamps, etc.).

3. **Batch Analysis with Apache Spark**:
   - Spark was used to perform trend detection, engagement metrics, geospatial analysis, and build network graphs.

4. **Simulated Real-time Streaming with Spark Streaming**:
   - Using Socket Server on `localhost:9999`, pre-collected data was streamed line-by-line.
   - Spark Streaming consumed the data and performed live analysis.

5. **Visualization with Streamlit**:
   - All analytics results were displayed on an interactive dashboard using Streamlit.
   - App runs locally on `localhost:8050`.
