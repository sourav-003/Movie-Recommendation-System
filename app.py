import gradio as gr
import pickle
import pandas as pd
import numpy as np

# Load the movie data and similarity matrix
new = pickle.load(open('movie_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

def movie_recommendation_interface(movie_title):
  """
  Recommends movies based on the input movie title using the pre-calculated similarity matrix.
  """
  try:
    index = new[new['title'] == movie_title].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommendations = [new.iloc[i[0]].title for i in distances[1:6]]
    return "Recommended Movies:\n" + "\n".join(recommendations)
  except IndexError:
    return "Movie not found. Please enter a valid movie title from the dataset."


# Create the Gradio interface
# Create the Gradio interface
iface = gr.Interface(
    fn=movie_recommendation_interface,
    inputs=gr.Textbox(label="Enter a movie title"),
    outputs=gr.Textbox(label="Recommendations"),
    title="Movie Recommendation System",
    description="Get movie recommendations based on the movie title you provide."
)

if __name__ == "__main__":
    iface.launch()
