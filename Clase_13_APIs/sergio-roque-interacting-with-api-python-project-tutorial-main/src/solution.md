# Interacting with the Spotify API

Spotify can be used as a data source for various data science projects. In this exercise, we will learn how to interact with the API of this social network. `Spotipy` is an open-source Python library that allows high-level use of the Spotify API.

## Step 1: Create a Spotify Developer Account

The first step is to create an application to be able to access Spotify API services. All the information can be found [here](https://developer.spotify.com/documentation/web-api).

Once you have logged in using your Spotify account, you can create the application to access the credentials needed to consume the API. You will need to fill in the following fields:

![Spotify create app](https://github.com/4GeeksAcademy/interacting-with-the-twitter-api-project-tutorial/blob/main/assets/spotify_1.PNG?raw=true)

> NOTE: As we are not going to use this API from any other web application, leave the **Redirect URI** field as `http://localhost/`.

Once you complete the form, you will have your application created. Next, in the **settings** section you can find your `Client ID` and `Client Secret`.

## Step 2: Initial configuration

- Create an `app.py` file inside the `./src/` folder.
- Install all the requirements from the `requirements.txt` file or just the required library.

```python
! pip install spotipy
```

    Collecting spotipy
      Downloading spotipy-2.23.0-py3-none-any.whl (29 kB)
    Collecting redis>=3.5.3 (from spotipy)
      Downloading redis-4.6.0-py3-none-any.whl (241 kB)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 241.1/241.1 kB 11.2 MB/s eta 0:00:00
    Requirement already satisfied: requests>=2.25.0 in /usr/local/python/3.10.8/lib/python3.10/site-packages (from spotipy) (2.27.1)
    Requirement already satisfied: six>=1.15.0 in /home/codespace/.local/lib/python3.10/site-packages (from spotipy) (1.16.0)
    Requirement already satisfied: urllib3>=1.26.0 in /usr/local/python/3.10.8/lib/python3.10/site-packages (from spotipy) (1.26.16)
    Collecting async-timeout>=4.0.2 (from redis>=3.5.3->spotipy)
      Downloading async_timeout-4.0.2-py3-none-any.whl (5.8 kB)
    Requirement already satisfied: certifi>=2017.4.17 in /home/codespace/.local/lib/python3.10/site-packages (from requests>=2.25.0->spotipy) (2023.5.7)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/python/3.10.8/lib/python3.10/site-packages (from requests>=2.25.0->spotipy) (2.0.12)
    Requirement already satisfied: idna<4,>=2.5 in /home/codespace/.local/lib/python3.10/site-packages (from requests>=2.25.0->spotipy) (3.4)
    Installing collected packages: async-timeout, redis, spotipy
    Successfully installed async-timeout-4.0.2 redis-4.6.0 spotipy-2.23.0

## Step 3: Environment variables

- Create a file `.env` on the root folder of the project.
- Insert both the `CLIENT_ID` and `CLIENT_SECRET` in the file, for example:

```py
CLIENT_ID="AAAAAAAAAAAABBBBBBBBBBBBCCCCCCCCCCCCC111111222222"
CLIENT_SECRET="DDDDDDDDDDDDDEEEEEEEEEEEEEEEEFFFFFFFFFFFFFF333333344444"
```

- Now we can insert this information into our Python program to start working with it

```py
import os
from dotenv import load_dotenv
load_dotenv()

client_id = os.environ.get('CLIENT_ID')
client_secret = os.environ.get('CLIENT_SECRET')
```

## Step 4: Initialize Spotipy library

After downloading the library and loading the environment variables, we can start working by initiating the API connection:

```py
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

con = spotipy.Spotify(auth_manager = SpotifyClientCredentials(client_id = client_id,
                                                              client_secret = client_secret))
```

Now we can work and interact with the API to perform the queries we want to perform.

## Step 5: Make API requests

In this case, I have chosen Drake. First, I get his ID by searching for his Spotify page and getting it from the URI. Now, I perform the search for his top 10 songs.

```py
artist_id = "3TVXtAsR1Inumwj472S9r4"

response = sp.artist_top_tracks("3TVXtAsR1Inumwj472S9r4")
if response:
  # We keep the "tracks" object of the answer
  tracks = response["tracks"]
  # We select, for each song, the data we are interested in and discard the rest
  tracks = [{k: (v/(1000*60))%60 if k == "duration_ms" else v for k, v in track.items() if k in ["name", "popularity", "duration_ms"]} for track in tracks]
```

## Step 6: Transform to Pandas DataFrame

Once we have modified the answer, we create the Pandas DataFrame from it:

```py
import pandas as pd

tracks_df = pd.DataFrame.from_records(tracks)
tracks_df.sort_values(["popularity"], inplace = True)

print(tracks_df.head(3))
```

## Step 7: Analyze statistical relationship

A scatter plot is a good alternative to determine visually and graphically whether two variables may or may not be related to each other:

```py
import seaborn as sns

scatter_plot = sns.scatterplot(data = tracks_df, x = "popularity", y = "duration_ms")
fig = scatter_plot.get_figure()
fig.savefig("scatter_plot.png")
```

![Scatter plot of popularity and duration of songs](https://github.com/4GeeksAcademy/interacting-with-the-twitter-api-project-tutorial/blob/main/assets/scatter_plot.png?raw=true)

As can be seen, there is no direct relationship between the length of the song and its popularity.
