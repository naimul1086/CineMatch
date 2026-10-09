# CineMatch — Content-Based Movie Recommendation System

A content-based movie recommender built on the **TMDB 5000 Movies** dataset. Give it a movie title (even misspelled, abbreviated, or partial) and it returns the most similar movies, ranked by content similarity and audience rating, with live posters, similarity scores, and genre filtering.

Built as a group project for **CSE4385 — Artificial Intelligence Lab**, Northern University Bangladesh.

![LOTR recommendations](images/demo_lotr.jpg)

---

## Features

- **Content-based filtering** using TF-IDF vectors and cosine similarity over plot, genres, keywords, cast, and director
- **Two interchangeable engines:** a pre-computed similarity matrix and a K-Nearest Neighbors model (`NearestNeighbors`, cosine metric) — both give near-identical neighbors
- **Quality-aware re-ranking** with the IMDB weighted-rating formula, so low-vote titles don't outrank well-established ones
- **Explainable output:** every recommendation shows its similarity %, rating, release date, genres, and director
- **Genre filtering** (e.g. only show Action results)
- **Flexible search:** case-insensitive, partial names, fuzzy matching for typos, and aliases like `LOTR`
- **Live posters** fetched from the TMDB API
- **Evaluation:** a Genre Overlap (Jaccard) score, used as a proxy metric because the dataset has no user-interaction ground truth

## How it works

| Step | What happens |
|------|--------------|
| 1 | Load `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`, merge on movie ID |
| 2 | Drop rows with a missing overview (3 rows → 4,800 movies remain) |
| 3 | Parse nested JSON columns: genres, keywords, top-3 cast, director |
| 4 | Combine everything into a single `tags` text field per movie |
| 5 | Lowercase, strip punctuation, apply Porter stemming |
| 6 | TF-IDF vectorization (5,000 features, English stop words removed) |
| 7 | Cosine similarity matrix (4,800 × 4,800), or KNN with cosine distance |
| 8 | Re-rank candidates with the weighted rating formula |
| 9 | Fetch posters from TMDB and render recommendation cards |

**Weighted rating:** `WR = (v / (v + m)) · R + (m / (v + m)) · C`
where `R` is the movie's average vote, `v` its vote count, `m` the 75th-percentile vote count, and `C` the mean vote across all movies.

## Results

Genre Overlap Score (Jaccard similarity between the query movie's genres and its top-5 recommendations):

| Query | Score |
|-------|-------|
| The Lord of the Rings: The Fellowship of the Ring | 0.63 |
| The Dark Knight Rises | 0.58 |
| Toy Story | 0.56 |
| Avatar | 0.40 |
| The Notebook | 0.38 |

![Genre overlap chart](images/genre_overlap_chart.png)

Franchise-heavy queries (LOTR, Batman) score highest because their neighbors share near-identical genre tags. Standalone romance/drama titles score lower because their metadata is less genre-distinctive.

<table>
  <tr>
    <td><img src="images/demo_dark_knight.jpg" alt="Dark Knight Rises recommendations"></td>
  </tr>
  <tr>
    <td><img src="images/demo_toy_story.jpg" alt="Toy Story recommendations"></td>
  </tr>
</table>

## Tech stack

Python · pandas · NumPy · scikit-learn · NLTK · Matplotlib · TMDB API · Google Colab

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/naimul1086/CineMatch.git
cd CineMatch
```

### 2. Get the dataset

Download the **TMDB 5000 Movie Dataset** from Kaggle:
<https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata>

You need these two files:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

Place them in a folder and update `dataset_path` in the *Load Dataset* cell of the notebook.

### 3. Get a TMDB API key

Create a free account at <https://www.themoviedb.org>, then request an API key under **Settings → API**.

In Google Colab, open the **Secrets** panel (key icon in the left sidebar), add a secret named `TMDB_API_KEY`, and enable notebook access. The notebook reads it with:

```python
from google.colab import userdata
TMDB_API_KEY = userdata.get('TMDB_API_KEY')
```

Never commit your API key to the repository.

### 4. Run the notebook

Open `CineMatch_Movie_Recommendation.ipynb` in Google Colab and run the cells from top to bottom. The last cells prompt for a movie name and an optional genre filter.

```
Enter movie name: lotr
Enter genre filter (press Enter to skip):
```

## Project structure

```
CineMatch/
├── CineMatch_Movie_Recommendation.ipynb   # full pipeline, step by step
├── images/                                # result screenshots used in this README
└── README.md
```

## Limitations

- **Cold start:** new movies with sparse metadata get weak recommendations
- **No personalization:** the same input always returns the same output
- **Metadata-dependent:** quality depends on how well a movie's overview and keywords were written
- **Proxy evaluation:** Genre Overlap measures thematic consistency, not user satisfaction

## Future work

- Hybrid filtering that blends in collaborative signals once user data is available
- Adjustable weighting between genre purity and rating quality
- A Streamlit or Gradio interface
- Approximate nearest neighbors (FAISS/Annoy) for much larger catalogs

## Team

| Name |
|------|
| Md Naimul Islam |
| Farjana Akter Urme |
| Tasfika Hasan Mahi |
| Amena Khanom Shopna |

**Course:** CSE4385 — Artificial Intelligence Lab, Northern University Bangladesh

## Acknowledgements

- [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) on Kaggle
- [The Movie Database (TMDB)](https://www.themoviedb.org) API for posters. This product uses the TMDB API but is not endorsed or certified by TMDB.
