from typing import Tuple, Any
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def visualize_data() -> None:
    sns.set(style="whitegrid")
    dataset = get_clean_data()
    plt.figure(figsize=(10, 6))
    sns.histplot(dataset['listeners_lastfm'], kde=True)
    plt.title('Listeners Score Distribution')
    plt.xlabel('Listeners Score')
    plt.ylabel('Count')
    st.pyplot(plt)
    plt.close()


def get_data_from_csv() -> pd.DataFrame:
    df = pd.read_csv('artists.csv', delimiter=',', low_memory=False)
    return df


def get_clean_data() -> pd.DataFrame:
    df = get_data_from_csv()
    df = df.dropna()
    return df


def preprocess_data() -> Tuple[Any, Any, Any, Any]:
    df = get_clean_data()
    X = df[['listeners_lastfm']]
    y = df['tags_lastfm']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def get_nearest_neighbors() -> Tuple[Any, Any]:
    X_train, X_test, y_train, y_test = preprocess_data()
    model = NearestNeighbors(n_neighbors=10, metric='cosine')
    model.fit(X_train)
    distances, indices = model.kneighbors(X_test)
    return distances, indices


def get_recommendations(user_input: str):
    distances, indices = get_nearest_neighbors()
    df = get_data_from_csv()
    artist_index = df[df['artist_lastfm'] == user_input].index[0]
    nearest_indices = indices[artist_index]
    recommendations = df.iloc[nearest_indices]
    return recommendations


def create_recommendation_dashboard() -> None:
    st.title('Recommendation Dashboard')
    user_input = st.text_input('Enter artist name')
    if st.button('Recommend'):
        if user_input:
            recommendations = get_recommendations(user_input)
            st.write(recommendations)
        else:
            st.write("Please enter an artist name.")


if __name__ == '__main__':
    st.sidebar.title('Music Recommendation System')
    menu = ['Home', 'Visualize Data', 'Recommendation Dashboard']
    choice = st.sidebar.selectbox('Select an option', menu)

    if choice == 'Home':
        st.title('Welcome to the Music Recommendation System')
        st.write('Use the sidebar to navigate through the app.')
    elif choice == 'Visualize Data':
        st.title('Data Visualization')
        visualize_data()
    elif choice == 'Recommendation Dashboard':
        create_recommendation_dashboard()
