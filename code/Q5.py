# 1
from wordcloud import WordCloud
from matplotlib import pyplot as plt

def generate_word_cloud(text_column, title):
    text = ' '.join(df[text_column].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {title}')

    # Save the Word Cloud as an image
    plt.savefig(f'{title}_wordcloud.png')

    # Show the plot
    plt.show()

for column in ['review_summary', 'review_advice', 'review_pros', 'review_cons']:
    generate_word_cloud(column, column.capitalize())

# 2
from textblob import TextBlob
from matplotlib import pyplot as plt
import seaborn as sns

def analyze_sentiment(text_column, title):
    df[f'{text_column}_sentiment'] = df[text_column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='job_category', y=f'{text_column}_sentiment', data=df)
    plt.title(f'Sentiment Analysis for {title}')

    # Save the plot as an image
    plt.savefig(f'{title}_sentiment_analysis.png')

    # Show the plot
    plt.show()

for column in ['review_summary', 'review_advice', 'review_pros', 'review_cons']:
    analyze_sentiment(column, column.capitalize())

# 3
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

def plot_top_words_for_topic(top_words, topic_num, title):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Word', y='Frequency', data=top_words, palette='viridis')
    plt.title(f'Top Words for Topic {topic_num} - {title}')
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    
    # Save the plot as an image
    plt.savefig(f'{title}_topic_{topic_num}.png')

    # Show the plot
    plt.show()

def topic_modeling(text_column, title):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(df[text_column])

    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)

    for i, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-5:][::-1]
        top_words = [{'Word': vectorizer.get_feature_names_out()[idx], 'Frequency': topic[idx]} for idx in top_words_idx]
        
        # Create a DataFrame for better visualization
        top_words_df = pd.DataFrame(top_words)
        
        # Plot and save the top words for each topic
        plot_top_words_for_topic(top_words_df, i + 1, title)

for column in ['review_summary', 'review_advice', 'review_pros', 'review_cons']:
    print(f"\nTopic Modeling for {column.capitalize()}:")
    topic_modeling(column, column.capitalize())

# 4
def analyze_review_lengths_binned(text_column, title, bin_width=10):
    df[f'{text_column}_length'] = df[text_column].apply(lambda x: len(str(x).split()))

    plt.figure(figsize=(10, 6))
    bins = range(0, df[f'{text_column}_length'].max() + bin_width, bin_width)
    sns.histplot(data=df, x=f'{text_column}_length', bins=bins, kde=True)
    plt.title(f'Distribution of Review Lengths for {title}')
    plt.xlabel('Review Length')
    plt.ylabel('Frequency')

    # Save the plot as an image
    plt.savefig(f'{title}_review_length_distribution_binned.png')

    # Show the plot
    plt.show()

for column in ['review_summary', 'review_advice', 'review_pros', 'review_cons']:
    analyze_review_lengths_binned(column, column.capitalize(), bin_width=10)


