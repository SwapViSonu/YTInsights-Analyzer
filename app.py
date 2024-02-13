import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
from pprint import pprint 
from dateutil import parser
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.ticker as ticker
import isodate
import base64
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from wordcloud import WordCloud




DEVELOPER_KEY = 'API KEY'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'


youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,developerKey=DEVELOPER_KEY)

def youtube_search(query, max_results=20):
  # Call the search.list method to retrieve results matching the specified
  # query term.
    search_response = youtube.search().list(
    q=query,
    part='id,snippet',
    maxResults=max_results).execute()
 
    channels = []
    channels_id=[]    
    
    for search_result in search_response.get('items', []):
      if search_result['id']['kind'] == 'youtube#channel':
        channels.append(search_result['snippet']['title'])
        channels_id.append(search_result['id']['channelId'])
    
    return channels_id


def get_channel_stats(youtube, channel_ids):
    
    """
    Get channel stats
    
    Params:
    ------
    youtube: build object of Youtube API
    channel_ids: list of channel IDs
    
    Returns:
    ------
    dataframe with all channel stats for each channel ID
    
    """
    
    all_data = []
    
    request = youtube.channels().list(
        part="snippet,contentDetails,statistics",
        id=','.join(channel_ids)
    )
    response = request.execute()

    # loop through items
    for item in response['items']:
        data = {'channelName': item['snippet']['title'],
                'playlistId': item['contentDetails']['relatedPlaylists']['uploads'],
                # 'channelId' : item['id'],
                'description': item['snippet']['description'],
                'subscribers': item['statistics']['subscriberCount'],
                'views': item['statistics']['viewCount'],
                'totalVideos': item['statistics']['videoCount'],
                'customUrl' : item['snippet']['customUrl'],
                'country' : item['snippet']['country'] if 'country' in item['snippet'] and item['snippet']['country'] else 'nan',
                'PublishDate':item['snippet']['publishedAt']
                
        }
        
        all_data.append(data)
        
    return pd.DataFrame(all_data)

def get_video_ids(youtube, playlist_id):
    
    video_ids = []
    
    request = youtube.playlistItems().list(
        part="snippet,contentDetails",
        playlistId=playlist_id,
        maxResults = 50
    )
    response = request.execute()
    
    for item in response['items']:
        video_ids.append(item['contentDetails']['videoId'])
        
    next_page_token = response.get('nextPageToken')
    while next_page_token is not None:
        request = youtube.playlistItems().list(
                    part='contentDetails',
                    playlistId = playlist_id,
                    maxResults = 50,
                    pageToken = next_page_token)
        response = request.execute()

        for item in response['items']:
            video_ids.append(item['contentDetails']['videoId'])

        next_page_token = response.get('nextPageToken')
        
    return video_ids
    
    
def get_video_details(youtube, video_ids):

    all_video_info = []
    
    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=','.join(video_ids[i:i+50])
        )
        response = request.execute() 

        for video in response['items']:
            stats_to_keep = {'snippet': ['channelTitle', 'title', 'description', 'tags', 'publishedAt'],
                             'statistics': ['viewCount', 'likeCount', 'favouriteCount', 'commentCount'],
                             'contentDetails': ['duration', 'definition', 'caption']
                            }
            video_info = {}
            video_info['video_id'] = video['id']

            for k in stats_to_keep.keys():
                for v in stats_to_keep[k]:
                    try:
                        video_info[v] = video[k][v]
                    except:
                        video_info[v] = None

            all_video_info.append(video_info)
    
    return pd.DataFrame(all_video_info)


#get video title using video ID
def get_video_title(youtube, video_id):
    try:
        response = youtube.videos().list(
            part='snippet',
            id=video_id
        ).execute()

        # Check if any items are returned
        if 'items' in response and len(response['items']) > 0:
            title = response['items'][0]['snippet']['title']
            return title
        else:
            return "Video not found."

    except HttpError as e:
        return f"Error: {e}"

def EDA(video_df):
    numeric_cols = ['viewCount', 'likeCount', 'favouriteCount', 'commentCount']
    video_df[numeric_cols] = video_df[numeric_cols].apply(pd.to_numeric, errors = 'coerce', axis = 1)
    video_df['durationSecs'] = video_df['duration'].apply(lambda x: isodate.parse_duration(x).total_seconds())
    video_df['publishedAt'] = video_df['publishedAt'].apply(lambda x: parser.parse(x)) 
    video_df['pushblishDayName'] = video_df['publishedAt'].apply(lambda x: x.strftime("%A"))  
    video_df['publishFullDate'] = video_df['publishedAt'].apply(lambda x: x.strftime("%Y-%m-%d"))
    video_df['publishYear'] = video_df['publishedAt'].apply(lambda x: x.strftime("%Y"))
    return video_df


#Scraping ALL Youtube Comments From Video_id
def comment_scrap(youtube, video_ids):
    request = youtube.commentThreads().list(part="snippet",videoId=video_ids ,maxResults=100)

    comments = []

    # Execute the request.
    response = request.execute()

    # Get the comments from the response.
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        public = item['snippet']['isPublic']
        comments.append([
            comment['authorDisplayName'],
          comment['publishedAt'],
          comment['likeCount'],
          comment['textOriginal'],
          comment['videoId'],
          public
      ])

    while (1 == 1):
        try:
            nextPageToken = response['nextPageToken']
        except KeyError:
            break
        nextPageToken = response['nextPageToken']
        # Create a new request object with the next page token.
        nextRequest = youtube.commentThreads().list(part="snippet", videoId=video_ids, maxResults=100, pageToken=nextPageToken)
        # Execute the next request.
        response = nextRequest.execute()
        # Get the comments from the next response.
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            public = item['snippet']['isPublic']
            comments.append([
                comment['authorDisplayName'],
                comment['publishedAt'],
                comment['likeCount'],
                comment['textOriginal'],
                comment['videoId'],
                public])

    # df2 = pd.DataFrame(comments, columns=['author', 'updated_at', 'like_count', 'text','video_id','public'])
    return pd.DataFrame(comments, columns=['author', 'updated_at', 'like_count', 'text','video_id','public'])


# Function to create a download link for a DataFrame
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
    href = f'<a href="data:file/csv;base64,{b64}" download="video_details.csv">Download CSV File</a>'
    return href


def analyze_dataframe(video_df):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title(f"Youtuber : {video_df['channelTitle'].unique()[0]}")
    
    total_likes = video_df['likeCount'].sum()/ 1e6 
    total_comments = video_df['commentCount'].sum()/ 1e6 
    total_views = video_df['viewCount'].sum()/ 1e6 
    # Streamlit UI
    
    # Calculate total likes, total comments, and total views in millions
    total_likes = video_df['likeCount'].sum() / 1e6  # Convert to millions
    total_comments = video_df['commentCount'].sum() / 1e6  # Convert to millions
    total_views = video_df['viewCount'].sum() / 1e6  # Convert to millions

    # Streamlit UI
    st.header("YouTube Video Analytics", divider='rainbow')

        # Arrange metrics side by side using st.columns
    col1, col2, col3 = st.columns(3)

    # Total Likes Tile
    delta_likes = (total_likes - (video_df['likeCount'].mean() / 1e6))
    col1.metric(label="Total Likes", value=f"{total_likes:.2f}M", delta=f"{delta_likes:.2f}M")

    # Total Comments Tile
    delta_comments = (total_comments - video_df['commentCount'].mean() / 1e6)
    col2.metric(label="Total Comments", value=f"{total_comments:.2f}M", delta=f"{delta_comments:.2f}M")

    # Total Views Tile
    delta_views = (total_views - video_df['viewCount'].mean() / 1e6)
    col3.metric(label="Total Views", value=f"{total_views:.2f}M", delta=f"{delta_views:.2f}M")
    
    st.subheader("Basic Statistical Values",divider='rainbow')
    pd.set_option('display.float_format', '{:.0f}'.format)
    st.write(video_df[['likeCount','commentCount','viewCount','durationSecs']].agg(['max',"min",'mean','sum','median']).T)
    # Display DataFrame
    # st.subheader("DataFrame Preview")
    # st.dataframe(dataframe)
    st.subheader("Analysis of Video Details",divider='rainbow')
    
    st.write("Top 5 Most Watched videos:")
    st.write(video_df.sort_values('viewCount',ascending=False).set_index("publishYear")[["title","viewCount","video_id"]][0:6])
    st.write("Top 5 Most Liked videos:")
    st.write(video_df.sort_values('likeCount',ascending=False).set_index("publishYear")[["title","likeCount","video_id"]][0:6])
    st.write("Top 5 Most Commented videos:")
    st.write(video_df.sort_values('commentCount',ascending=False).set_index("publishYear")[["title","commentCount","video_id"]][0:6])
    
    
    #grouped_df by year by sum of Numerical columns
    grouped_df =  video_df.groupby('publishYear').agg({'viewCount': 'sum', 'likeCount': 'sum','durationSecs':'sum'}).reset_index()
    # View Count Distribution
    # Create subplots with 1 row and 2 columns
    fig = make_subplots(rows=2, cols=1, subplot_titles=['Total View Count and Like Count', 'Total Durations of Video in Sec'])
    fig.add_trace(go.Bar(x=grouped_df['publishYear'], y=grouped_df['viewCount'], name='View Count', marker_color='blue'),row=1, col=1)
    fig.add_trace(go.Bar(x=grouped_df['publishYear'], y=grouped_df['likeCount'], name='Like Count', marker_color='green'),row=1, col=1)
    fig.add_trace(go.Bar(x=grouped_df['publishYear'], y=grouped_df['durationSecs'], name='Duration (secs)', marker_color='purple'),row=2, col=1)
    fig.update_layout(title_text='Total View Count, Like Count, and Durations by Publish Year', showlegend=True)
    st.plotly_chart(fig)
    
    # Create subplots with 1 row and 3 columns
    fig = make_subplots(rows=1, cols=3, subplot_titles=['Like Count', 'Comment Count', 'View Count'])
    # Add distribution plots for each variable
    fig.add_trace(go.Histogram(x=video_df['likeCount'], name='Like Count'), row=1, col=1)
    fig.add_trace(go.Histogram(x=video_df['commentCount'], name='Comment Count'), row=1, col=2)
    fig.add_trace(go.Histogram(x=video_df['viewCount'], name='View Count'), row=1, col=3)
    fig.update_layout(title_text='Distribution of Like Count, Comment Count, and View Count', showlegend=True)
    st.plotly_chart(fig)
    
    
    st.subheader('Line Plot',divider='rainbow')
    fig = px.line(video_df, x="publishFullDate", y="viewCount", title='Variations of view with time')
    st.plotly_chart(fig)
    
    fig = px.line(video_df, x="publishFullDate", y="likeCount", title='Variations of Likes with time')
    st.plotly_chart(fig)

    st.subheader('Scatter Plot Relationship',divider='rainbow')
    fig = px.scatter(video_df,x='likeCount', y='commentCount',  trendline="ols",title=' Like count vs Comment count')
    st.plotly_chart(fig)    
    
    fig = px.scatter(video_df,x='viewCount', y='likeCount',  trendline="ols",title=' View count vs Like count')
    st.plotly_chart(fig)  
    
    fig = px.scatter(video_df,x='viewCount', y='commentCount',  trendline="ols",title=' View count vs comment count')
    st.plotly_chart(fig)  
    
    
    st.subheader('Year Wise Video Count',divider='rainbow')
    df = video_df.groupby(by=["publishYear"]).size().reset_index(name="counts")
    fig= px.bar(data_frame=df, x="publishYear", y="counts",color_discrete_sequence=['coral'], barmode="group")
    st.plotly_chart(fig)
    
    #which month most video getting Uploaded
    df = video_df.groupby(by=["pushblishDayName"]).size().reset_index(name="counts")
    fig= px.bar(data_frame=df, x="pushblishDayName", y="counts",color_discrete_sequence=['aquamarine'], barmode="group")
    st.subheader('In which day most video getting Uploaded',divider='rainbow')
    st.plotly_chart(fig) 
    
    stop_words = set(stopwords.words('english'))
    video_df['title_no_stopwords'] = video_df['title'].apply(lambda x: [item for item in str(x).split() if item not in stop_words])
    all_words = list([a for b in video_df['title_no_stopwords'].tolist() for a in b])
    all_words_str = ' '.join(all_words) 
    def plot_cloud(wordcloud):
        plt.figure(figsize=(30, 20))
        plt.imshow(wordcloud) 
        plt.axis("off");

    wordcloud = WordCloud(width = 2000, height = 1000, random_state=1, background_color='black', 
                      colormap='Set2', collocations=False).generate(all_words_str)
    plot_cloud(wordcloud)

    st.subheader('Most Appeared Word in video titles',divider='rainbow')
    st.pyplot()






# Function to analyze YouTube video for a given YouTuber
# Streamlit UI
def main():
    st.title("YouTube Analysis App")

    # Tabs for different functions
    tabs = ["Search & Channel Stats", "Video IDs & Details", "Comment Scrap"]
    selected_tab = st.sidebar.radio("Select Tab", tabs)

    if selected_tab == "Search & Channel Stats":
        st.header("YouTube Search & Channel Stats")
        # Input: YouTuber's username
        youtuber_query = st.text_input("Enter YouTuber's query:")
        if st.button("Search & Get Channel Stats"):
            if youtuber_query:
                # Perform YouTube search and get channel stats
                channel_ids = youtube_search(youtuber_query)
                if channel_ids:
                    channel_stats =get_channel_stats(youtube, channel_ids)
                    st.dataframe(channel_stats)
                else:
                    st.warning("No channels found.")
            else:
                st.warning("Please enter a YouTuber's query.")

    elif selected_tab == "Video IDs & Details":
        st.header("YouTube Video IDs & Details")
        # Input: Playlist ID
        playlist_id = st.text_input("Enter Playlist ID:")
        if st.button("Get Video IDs & Details"):
            if playlist_id:
                # Get video IDs and details
            
                video_ids = get_video_ids(youtube, playlist_id)
                if video_ids:
                    video_details = EDA(get_video_details(youtube, video_ids))
                    st.subheader("List of Videos Enriched with Informative Content")
                    st.dataframe(video_details)
                     # Download button for DataFrame
                    st.markdown(get_table_download_link(video_details), unsafe_allow_html=True)

                    # Analysis of DataFrame
                    # st.subheader("Analysis of Video Details")
                    analyze_dataframe(video_details)


                    # st.subheader("Likes vs. Dislikes")
                    # fig= sns.scatterplot(x='likeCount', y='commentCount', data=video_details)
                    # st.pyplot(fig)
                    
                
                else:
                    st.warning("No videos found in the specified playlist.")
            else:
                st.warning("Please enter a Playlist ID.")

    elif selected_tab == "Comment Scrap":
        st.header("YouTube Comment Scrap")
        # Input: Video IDs
        video_ids_input = st.text_input("Enter Video IDs (comma-separated):")
        # video_ids = video_ids_input.split(",") if video_ids_input else []
        if st.button("Scrap Comments"):
            if video_ids_input:
                # Scrap comments
                video_title = get_video_title(youtube, video_ids_input)
                st.header(f"Video Title: {video_title}",divider='rainbow')
                comments_df = comment_scrap(youtube, video_ids_input)
                st.dataframe(comments_df)
                st.markdown(get_table_download_link(comments_df), unsafe_allow_html=True)
                
                st.header('Most Liked comment',divider='rainbow') 
                st.write(comments_df.sort_values(by='like_count', ascending=False).set_index('like_count')[['text','author']][0:10])
                
                
                stop_words = set(stopwords.words('english'))
                comments_df['text_no_stopwords'] = comments_df['text'].apply(lambda x: [item for item in str(x).split() if item not in stop_words])
                all_words = list([a for b in comments_df['text_no_stopwords'].tolist() for a in b])
                all_words_str = ' '.join(all_words) 
                def plot_cloud(wordcloud):
                    plt.figure(figsize=(30, 20))
                    plt.imshow(wordcloud) 
                    plt.axis("off");
                wordcloud = WordCloud(width = 2000, height = 1000, random_state=1, background_color='white', 
                      colormap='Set2', collocations=False).generate(all_words_str)
                plot_cloud(wordcloud)
                st.header('Most Appeared Word in video comment',divider='rainbow') 
                st.pyplot()

            else:
                st.warning("Please enter Video IDs (comma-separated).")

if __name__ == "__main__":
    main()
