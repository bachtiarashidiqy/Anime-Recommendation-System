# Anime Recommendation System Using Content-Based Filtering and K-Nearest Neighbor

## Project Domain
This project falls within the field of Recommender Systems and Data Science, focusing specifically on delivering personalized entertainment content recommendations. It employs methodologies from machine learning, data analysis, and information retrieval to improve user engagement by suggesting anime tailored to individual preferences.

## Project Overview
Anime, a distinctive form of Japanese animation, has gained widespread popularity across various streaming platforms. However, the vast array of available anime titles often makes it challenging for users to discover content that aligns with their tastes. This research proposes a recommendation system designed to address this issue. The system analyzes users’ viewing histories, favorite genres, and ratings to generate personalized anime suggestions. Additionally, factors such as trending titles, user reviews, and community recommendations are incorporated to enhance the relevance of the suggestions. 

This approach offers significant benefits for both users and streaming providers. Users can effortlessly find new favorite anime, explore different genres, and discover titles that match their mood. For streaming services, the system can increase viewer engagement, diversify content offerings, improve user satisfaction, and gain insights into user preferences. Ultimately, an effective recommendation system can serve as a valuable tool in helping users find anime that suits their tastes and elevates their overall viewing experience [1].

## Business Understanding
Developing an anime recommendation system holds the potential to significantly enhance user experience and platform performance. By making it easier for users to find anime aligned with their interests, platforms can boost user engagement, satisfaction, and retention. Furthermore, such a system can facilitate content discovery, support the platform’s goal of offering diverse and relevant content, and enable better understanding of user preferences, which can inform content acquisition and marketing strategies [2].

### Problem Statements
- How can we develop an anime recommendation system that suggests titles based on the user’s preferred genres?
- Using user rating data, how can streaming platforms recommend anime that users have not yet watched?
- How can we build recommendation models using Cosine Similarity and K-Nearest Neighbor algorithms?
- How can we evaluate the performance of the developed recommendation models?

### Goals
The project aims to:

- Generate top-N anime recommendations for users based on genre preferences.
- Provide multiple relevant anime suggestions that align with user preferences and are yet to be viewed.
- Develop recommendation models utilizing Cosine Similarity and K-Nearest Neighbor techniques based on selected features from the dataset.
- Assess the effectiveness of the models using appropriate evaluation metrics.

### Solution Approach
The approach involves conducting Exploratory Data Analysis (EDA) and creating visualizations to understand the data better. To ensure the development of accurate models, data cleaning procedures are implemented, including removing missing values, checking for duplicates, eliminating alphanumeric symbols, and removing URLs. One-hot encoding is applied to convert categorical data into numerical form. To evaluate model performance, metrics such as Precision, Calinski-Harabasz Score, and Davies-Bouldin Score are used, ensuring the models are both effective and reliable.

## Data Understanding
### Variable Description

| Type | Description |
|----------|-----------------------------------------------------------|
| Title | Anime Dataset 2023 |
| Source |[Kaggle](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset/data?select=anime-filtered.csv) |
| Maintainer |[Sajid](https://www.kaggle.com/dbdmobile) |
| License | Database: Open Database, Contents: Database Contents |
| Visibility | Public |
| Tags | Arts and Entertainment, Movies and TV Shows, Anime and Manga, Popular Culture, Japan |
| Usability | 10.00 |

This dataset is compiled from the platform [MyAnimeList](https://myanimelist.net/), a popular online community and database for anime and manga enthusiasts. The platform offers valuable information about anime titles, user profiles, and user ratings across various anime. The dataset used in this project is publicly available on Kaggle under the name Anime Dataset 2023. It can be downloaded from Kaggle via this link: [Anime Dataset 2023](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset/data?select=anime-filtered.csv).

**Here's the information on the datasets**:
 - Datasets are csv (Comma-Seperated Values) files.
 - The dataset is in the form of 6 CSV files, namely: 
    * anime-dataset-2023.csv  
    * anime-filtered.csv      
    * final_animedataset.csv  
    * users-filtered.csv       
    * users-details-2023.csv  
    * users-score-2023.csv

In this model, the dataset used is the `anime-filtered.csv` file.
 - The dataset has 14952 samples with 25 features.
 - The dataset has 15 `object` features, 8 `int64` features, and 2 `float64` features.
 - There are *Missing value* in `sypnopsis` feature 1350 and Ranked feature 1721`.
 - There is no duplicate data.
  
### Variable Descriptions
The anime datasets column has the following information:
**`anime_id`:** A unique ID for each anime (number or identifier code).
**`Name`:** The title of the anime in its original language.
**`Score`:** The score or rating given to the anime.
**`Genres`:** Genre of the anime, separated by commas (for example, Action, Comedy, Fantasy).
**`English name`:** The English title of the anime (if available).
**`Japanese name`:** Anime title in Japanese (if available).
**`Synopsis`:** A short description or plot summary of the anime.
**`Type`:** Type of anime (for example, TV Series, Movie, OVA, etc.).
**`Episodes`:** Number of episodes in the anime.
**`Aired`:** The date the anime was aired.
**`Premiered`:** The season and year the anime premiered.
**`Producers`:** The production company or producer of the anime.
**`Licensors`:** Parties that have licensed the anime (for example, streaming platforms).
**`Studios`:** Animation studios that work on anime.
**`Source`:** The source material of the anime (for example, manga, light novel, original).
**`Duration`:** The duration of each anime episode.
**`Rating`:** Age restriction for watching anime.
**`Ranked`:** Ranking of anime based on popularity or other criteria.
**`Popularity`:** Anime popularity ranking.
**`Members`:** Number of members who have added anime to their list on the platform.
**`Favorites`:** Number of users who marked the anime as favorite.
**`Watching`:** The number of anime that are currently being watched by users.
**`Completed`:** The number of anime that the user has finished watching.
**`On Hold`:** The number of anime that the user has put on hold.
**`Dropped`:** The number of anime stopped by the user.


Figures:
![Averages Rating](https://github.com/user-attachments/assets/6e5aaa1d-4ddb-4105-ba73-6004fe2ec777)

Figure 1: Distribution of ratings:

The average rating is approximately 6.5, with the minimum at 1.8 and the maximum at 9.1.

![Categories Distribution](https://github.com/user-attachments/assets/cb51bcd7-c30c-4c5e-920b-ed1eea87b075)

Figure 2: Distribution of anime categories:

Anime types include TV, OVA, Movie, Special, ONA, and Music. TV (Television Series) are broadcast on TV with varying episodes; OVA (Original Video Animation) are released for home media; Movies are theatrical releases; Specials are bonus content; ONA (Original Net Animation) are online distributed; Music is created for album or single releases.

![Top10Community](https://github.com/user-attachments/assets/76b13dde-dfd1-476e-8bc7-1fa3a03e362a)

Figure 3: Top 10 Anime Communities:

The most active community is for "Death Note," followed by "Shingeki no Kyojin," "Fullmetal Alchemist: Brotherhood," "Sword Art Online," "One Punch Man," "Boku no Hero Academia," "Tokyo Ghoul," "Naruto," "Steins Gate," and "No Game No Life." These community sizes indicate the popularity of these titles among users, providing insights for recommendation development.

![Top10AnimeRating](https://github.com/user-attachments/assets/266c9b60-c1d7-4f0a-89d8-ec1d7124ed61)

Figure 4: Top 10 Highest Rated Anime:

The highest-rated anime is "Fullmetal Alchemist: Brotherhood," followed by "Shingeki no Kyojin: Final Season," "Steins Gate," "Shingeki no Kyojin Season 3 Part 2," "Hunter x Hunter (2011)," "Gintama°," "Gintama'," "Ginga Eiyuu Densetsu," "Gintama: Enchousen," and "3-gatsu no Lion 2nd Season." Ratings reflect user contributions, with higher ratings indicating greater popularity among viewers.

**Data Preparation**

During the Data Preparation phase, text cleaning was performed to remove punctuation marks and URLs from the dataset. For handling missing values, the approach used was *dropping* the affected records via the `drop()` method. The reason for choosing this method is that the omitted data does not significantly impact the model’s performance. Initially, the dataset contained 14,952 records; after removing entries with missing values, the dataset size was reduced to 13,229 records. 

For building the recommendation system in this project, the features utilized were `Name`, `Score`, `Genres`, `Type`, and `Studios`. The genre-based recommendation system primarily relies on the `Name` and `Genres` attributes, while the collaborative filtering approach uses `Name`, `Score`, and `Type`. Additionally, one-hot encoding was applied to transform the categorical features `Type` and `Score` into a format that is more easily processed by machine learning models.

### Modeling

This project employed only two algorithms: Cosine Similarity and K-Nearest Neighbors. Both algorithms are used to analyze the similarity between data points based on their features.

### Cosine Similarity

Cosine similarity is a method to measure how similar two vectors are within a multi-dimensional space[3]. It calculates the cosine of the angle between two vectors, which represent data points or features. The similarity score ranges from -1 to 1, where:
- 1 indicates the vectors are perfectly aligned (completely similar),
- 0 indicates the vectors are orthogonal (no relation),
- -1 indicates the vectors are diametrically opposed (completely dissimilar).

This method is frequently used in text processing and clustering tasks to determine the degree of similarity between documents or feature vectors in a dataset. 

The formula for cosine similarity is:

$$
Cosine\ Similarity (A, B) = \frac{A \cdot B}{||A|| \times ||B||}
$$

where:
- \(A \cdot B\) is the dot product of vectors A and B,
- \(||A||\) is the Euclidean norm (magnitude) of vector A,
- \(||B||\) is the Euclidean norm (magnitude) of vector B.

To evaluate the model, the following code snippet was used:

```python
anime_recommendations('One Piece')
```

| Name | Genres |
|--------------------------------------------------|--------------------------------------------------------------|
| One Piece Episode of Sorajima | Action, Adventure, Comedy, Drama, Fantasy, Shounen, Super Power. |
| One Piece Episode of Merry Mou Hitori no Nakama no Monogatari | Action, Adventure, Comedy, Drama, Fantasy, Shounen, Super Power. |
| One Piece Movie 14 Stampede | Action, Adventure, Comedy, Drama, Fantasy, Shounen, Super Power. |
| One Piece Episode of East Blue Luffy to 4 nin no Nakama no Daibouken | Action, Adventure, Comedy, Drama, Fantasy, Shounen, Super Power. |
| One Piece Episode of Sabo 3 Kyoudai no Kizuna Kiseki no Saikai to Uketsugareru Ishi | Action, Adventure, Comedy, Drama, Fantasy, Shounen, Super Power. |

**Table 1: Results of Content-Based Filtering Model Testing (with Genre Filter).**

Based on **Table 1**, the content-based filtering system successfully recommended the top 5% of anime similar to **One Piece**, including several films and series from the **One Piece** franchise itself. This indicates that if a user enjoys **One Piece**, the system can suggest other series or movies within the **One Piece** universe. By focusing on genre similarities, the system identifies anime with comparable themes, enabling users to discover content aligned with their preferences based on their interest in **One Piece**.

**Advantages of Cosine Similarity:**
- Low computational complexity, making it efficient.
- Suitable for high-dimensional datasets as it is unaffected by the number of dimensions.

**Disadvantages of Cosine Similarity:**
- Considers only the direction of vectors, not their magnitude.
- Differences in vector magnitude are not fully accounted for, which can lead to similar similarity scores even for vectors with vastly different lengths if their directions are aligned.

### K-Nearest Neighbor
**K-Nearest Neighbor (KNN)** is one of the simplest algorithms used for data classification. This algorithm is easy to understand because it groups data based on proximity to neighboring data points [4]. In KNN, a certain number of nearest neighbors (determined by the parameter K) are considered to decide the class or label of the data to be classified. When K=1, the algorithm only considers the single closest neighbor or data record with the most similar characteristics.
The KNN algorithm is expressed with the following formula:

$$
Euclidean\ Distance (P, Q) = \sqrt{\sum (P_i - Q_i)^2}
$$

where:
- \( P_i \) represents the i-th feature of data point P.
- \( Q_i \) represents the i-th feature of data point Q (from data set D).
- \( \sum \) denotes the summation over all features.

Below are the results of testing the K-Nearest Neighbor model using the Euclidean Distance metric:

If a user likes the application: **Neon Genesis Evangelion Death Rebirth**, then the following applications are also likely to be enjoyed:

| Anime Name | Similarity Score |
|--------------|------------------|
| Neon Genesis Evangelion Death Rebirth | 100.0% |
| Neon Genesis Evangelion The End of Evangelion | 98.94% |
| Kekkaishi TV | 98.59% |
| Doraemon Doraemon Comes Back | 98.59% |
| Dr Slump Aralechan | 98.59% |

**Table 2: Results of K-Nearest Neighbor Model Testing**

Based on **Table 2**, the K-Nearest Neighbor model provides anime recommendations based on similarity in features such as 'Name', 'Score', 'Type', and 'Studios'. The recommendations for anime similar to **Neon Genesis Evangelion Death Rebirth** include: *Neon Genesis Evangelion Death Rebirth*, *Neon Genesis Evangelion The End of Evangelion*, *Kekkaishi TV*, *Doraemon Doraemon Comes Back*, and *Dr Slump Aralechan*. The similarity percentages are 100.0%, 98.94%, 98.59%, 98.59%, and 98.59%, respectively. Clearly, this model can be very helpful for users in finding content similar to **Neon Genesis Evangelion Death Rebirth**.

**Advantages of KNN:**
- Very fast training process.
- Simple and easy to understand.
- Resistant to noise in training data.
- Effective with large training datasets.

**Disadvantages of KNN:**
- The choice of the value of K can introduce bias.
- Computationally intensive, especially with large or high-dimensional datasets.
- Requires storing all training data, leading to high memory usage.
- Sensitive to irrelevant features that may affect classification accuracy.

**Evaluation**

Evaluation metrics are used to assess how well a model performs. In this context, several common evaluation metrics are employed to measure the model's effectiveness, including Precision, Calinski-Harabasz Score, and Davies-Bouldin Score. These metrics aim to provide insights into how effectively the model performs specific tasks such as classification or clustering.

### Precision
**Precision** is a key metric for evaluating clustering performance. It helps measure how accurately the model identifies positive data points. A high precision indicates that the model rarely makes false positive predictions, making its positive predictions more reliable.

Precision is calculated with the following formula:

$$
Precision = \frac{TP}{TP + FP}
$$

where:
- **TP** (True Positive): The number of data points correctly predicted as positive.
- **FP** (False Positive): The number of data points incorrectly predicted as positive.

**Interpretation**: Based on _Table 1: Results of Content-Based Filtering Model Testing (with Genre Filter)_, the precision of the top-5 recommendation model is perfect, at 5/5 or 100%. This indicates that the model provides highly accurate recommendations, correctly suggesting anime with similar names and genres to **One Piece**, such as Action, Adventure, Comedy, Drama, Fantasy, Shounen, and Super Power. The top five recommended applications share genres closely aligned with **One Piece**.

### Calinski-Harabasz Score
The **Calinski-Harabasz (CH) score** evaluates clustering algorithms by measuring how well the data points are grouped into distinct and compact clusters [5]. It is defined as the ratio between the within-cluster scatter and the between-cluster scatter. A higher CH score signifies better clustering performance, without requiring any prior knowledge of true labels.

The formula for the Calinski-Harabasz score is:

$$
CH = \frac{B}{W} \times \frac{N - k}{k - 1}
$$

where:
- \( B \) is the between-cluster scatter.
- \( W \) is the within-cluster scatter.
- \( N \) is the total number of data points.
- \( k \) is the number of clusters.

The model testing was performed using the following code:
```python
calinski_harabasz_score(data_new, animedf_name)
```
which yielded a score of:
```
3.1613291729405617
```

**Interpretation**: The relatively low CH score suggests that the clusters in this model are not well separated, indicating potential issues with recommendation accuracy for some applications. This highlights the need for further review or adjustment of the model to improve cluster separation and recommendation quality.

### Davies-Bouldin Score
The **Davies-Bouldin (DB) Score** assesses clustering performance by measuring the average similarity between each cluster and its most similar one, comparing intra-cluster distances to inter-cluster distances [5]. With a minimum score of zero, a lower DB score indicates better clustering, reflecting clusters that are close together and less dispersed. Unlike some other metrics, DB does not require prior knowledge of true labels, similar to the Silhouette Score, but with a simpler formulation for efficient evaluation without additional data insights.

The formula for the Davies-Bouldin score is:

$$
DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{R_i + R_j}{d(c_i, c_j)} \right)
$$

where:
- \( k \) is the number of clusters.
- \( R_i \) is the radius of cluster \( i \).
- \( d(c_i, c_j) \) is the distance between the centers of clusters \( i \) and \( j \).

The score is calculated as the average of the ratios between intra-cluster radius and inter-cluster center distances, evaluating the similarity among clusters.

The model evaluation was performed using:
```python
davies_bouldin_score(data_new, animedf_name)
```
which produced a score of:
```
0.7864266764751376
```

**Interpretation**: The relatively low Davies-Bouldin score indicates that the clustering in this model has a good level of separation, leading to effective recommendations. This suggests that the clustering approach is sufficiently capable of distinguishing different data groups, resulting in quality recommendations for anime applications.
 
 
## Referensi
1. Liu, J. Y. (2018, September). A survey of deep learning approaches for recommendation systems. In Journal of Physics: Conference Series (Vol. 1087, No. 6, p. 062022). IOP Publishing.
2. Khatwani, S., & Chandak, M. B. (2016, September). Building personalized and non personalized recommendation systems. In 2016 International Conference on Automatic Control and Dynamic Optimization Techniques (ICACDOT) (pp. 623-628). IEEE.
3. A. R. Lahitani, A. E. Permanasari and N. A. Setiawan, "Cosine similarity to determine similarity measure: Study case in online essay assessment," 2016 4th International Conference on Cyber and IT Service Management, Bandung, Indonesia, 2016, pp. 1-6, doi: 10.1109/CITSM.2016.7577578.
4. K. Taunk, S. De, S. Verma and A. Swetapadma, "A Brief Review of Nearest Neighbor Algorithm for Learning and Classification," 2019 International Conference on Intelligent Computing and Control Systems (ICCS), Madurai, India, 2019, pp. 1255-1260, doi: 10.1109/ICCS45141.2019.9065747.
5. Gagolewski, M., Bartoszuk, M., & Cena, A. (2021). Are cluster validity measures (in) valid? Information Sciences, 581, 620–636. https://doi.org/10.1016/j.ins.2021.10.004.
