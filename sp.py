import socket
import time
import pandas as pd

df = pd.read_csv("C:/Users/Bhargavi/Desktop/processed_tweets.csv")

HOST = "localhost"
PORT = 9999


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"Listening on {HOST}:{PORT}")
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        for _, row in df.iterrows():
            tweet = str(row["Text"]) 
            conn.sendall((tweet + "\n").encode("utf-8"))
            time.sleep(1)  
