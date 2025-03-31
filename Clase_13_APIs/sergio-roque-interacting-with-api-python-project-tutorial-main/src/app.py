import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# load the .env file variables
load_dotenv()

client_id = os.environ.get("CLIENT_ID")
client_secret = os.environ.get("CLIENT_SECRET")

spotify = spotipy.Spotify(auth_manager = SpotifyClientCredentials(client_id = client_id,
                                                              client_secret = client_secret))

artist = "05fG473iIaoy82BF1aGhL8"
results = spotify.artist_top_tracks(artist_id=artist)

data = {"Name":[], "Popularity":[], "Duration (min)":[]}
ccounter = 0

for track in results['tracks'][:10]:
    data["Name"].append(track["name"])
    data["Popularity"].append(track["popularity"])
    data["Duration (min)"].append(round(track["duration_ms"]/(1000*60), 2))
    # Extraer nombre de artista
    if ccounter == 0:
        artist = track["artists"][0]["name"]
        ccounter += 1

df = pd.DataFrame(data)
df_sorted = df.sort_values(by="Popularity", ascending=False)
print(f"Top 3 de canciones mas populares de {artist}:")
print(df_sorted.head(3))

plt.scatter(df_sorted['Duration (min)'], df_sorted['Popularity'])
plt.xlabel('Duración (minutos)')
plt.ylabel('Popularidad')
plt.title('Popularidad vs Duración de Canciones')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("assets/scatter.png", dpi=300)

print("CONCLUSIONES:")
print("El grafico no me evidencia ningun patron entre duracion de canciones y popularidad,"
" probablemente dependa de otros factores y no de la duracion en si de las canciones")