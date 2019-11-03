import csv
import os
import pickle
import re
import traceback
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Optional
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pyLDAvis
import pytz
import requests
import seaborn as sns
from bs4 import BeautifulSoup, Comment
from pandas import DataFrame
from pyLDAvis import sklearn as sklearn_lda
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud


class NewsDataClassifier:
    @staticmethod
    def clean_data(data: DataFrame) -> DataFrame:
        """
        remove punctuation and lower case
        remove number
        :param data:
        :return:
        """
        # remove punctuation
        data["news_data"] = data["news_data"].map(lambda x: re.sub(r'[^\w\s]', '', x))
        data["description"] = data["description"].map(lambda x: re.sub(r'[^\w\s]', '', x))
        # convert title into lowercase
        data["news_data"] = data["news_data"].map(lambda x: x.lower())
        # remove number
        data["news_data"] = data["news_data"].replace("\d+", '', regex=True)

        return data[data["news_data"] != ""]

    @staticmethod
    def create_word_cloud(words_string: str, pic_name=None):
        """
        create word-cloud photo
        :param words_string:
        :param pic_name:
        :return:
        """
        # Create a WordCloud object
        wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color="steelblue")

        # Generate a word cloud
        wordcloud.generate(words_string)

        # Visualize the word cloud
        # wordcloud.to_image()

        # save wordcloud
        wordcloud.to_file(pic_name + ".png") if pic_name else wordcloud.to_file("wordcloud.png")

    @staticmethod
    def plot_10_most_common_words(count_data, count_vectorizer, pic_name=None):
        """
        # Helper function for plot common words
        :param count_data:
        :param count_vectorizer:
        :param pic_name:
        :return:
        """
        sns.set_style('whitegrid')
        words = count_vectorizer.get_feature_names()
        total_counts = np.zeros(len(words))
        for t in count_data:
            total_counts += t.toarray()[0]

        count_dict = (zip(words, total_counts))
        count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
        words = [w[0] for w in count_dict]
        counts = [w[1] for w in count_dict]
        x_pos = np.arange(len(words))

        plt.figure(2, figsize=(15, 15 / 1.6180))
        plt.subplot(title="10 most common words")
        sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
        sns.barplot(x_pos, counts, palette='husl')
        plt.xticks(x_pos, words, rotation=90)
        plt.xlabel("words")
        plt.ylabel("counts")
        # plt.show()
        plt.savefig("common_wordcount.png") if pic_name is None else plt.savefig(pic_name + ".png")

    def lda_model_training(self, count_data, count_vectorizer):
        """
        lda model training
        check the topic most common words
        :return:
        """
        warnings.simplefilter("ignore", DeprecationWarning)
        # Helper function

        def print_topics(model, count_vectorizer, n_top_words):
            words = count_vectorizer.get_feature_names()
            for topic_idx, topic in enumerate(model.components_):
                print("\nTopic #%d:" % topic_idx)
                print(" ".join([words[i]
                                for i in topic.argsort()[:-n_top_words - 1:-1]]))

        # Tweak the two parameters below (use int values below 15)
        number_topics = 5
        number_words = 10

        # Create and fit the LDA model
        lda = LDA(n_components=number_topics)
        lda.fit(count_data)

        #create a html using ldavis
        self.lda_model_checking(lda, count_data, count_vectorizer, number_topics)

        # Print the topics found by the LDA model
        print("Topics found via LDA:")
        print_topics(lda, count_vectorizer, number_words)

    @staticmethod
    def lda_model_checking(lda, count_data, count_vectorizer, number_topics, lda_html_name=None):
        """
        draw the topic distribution by group using ldavis library
        :param count_data:
        :param count_vectorizer:
        :param lda_html_name:
        :return:
        """
        if lda_html_name is None:
            lda_html_name = "./ldavis_prepared_"
        LDAvis_data_filepath = os.path.join(lda_html_name + str(number_topics))
        # # this is a bit time consuming - make the if statement True
        # # if you want to execute visualization prep yourself
        if 1 == 1:
            LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)

        # load the pre-prepared pyLDAvis data from disk
        with open(LDAvis_data_filepath, 'rb') as f:
            LDAvis_prepared = pickle.load(f)
        pyLDAvis.save_html(LDAvis_prepared, lda_html_name + str(number_topics) + '.html')

    def get_max_topic_score(self, clean_data: DataFrame, number_topics: int, pic_name=None) -> DataFrame:
        """
        get group by document using lda
        :param clean_data:
        :param number_topics:
        :param pic_name:
        :return:
        """
        # Initialise the count vectorizer with the English stop words
        count_vectorizer = CountVectorizer(stop_words="english")
        count_data = count_vectorizer.fit_transform(clean_data["news_data"])

        lda = LDA(n_components=number_topics, random_state=0)
        lda.fit(count_data)
        # get topic_score
        topic_score = lda.fit_transform(count_data)

        # create html using LDAvis
        self.lda_model_checking(lda, count_data, count_vectorizer, number_topics, pic_name)

        # Visualise the 10 most common words
        classifier.plot_10_most_common_words(count_data, count_vectorizer, pic_name)

        # join all description text and create word cloud figure
        classifier.create_word_cloud(",".join(list(clean_news["description"].values)))

        clean_data["max_score"] = np.amax(topic_score, axis=1)
        clean_data["max_score_index"] = np.argmax(topic_score, axis=1)
        return clean_data

    @staticmethod
    def get_histograms(results: DataFrame, topics_num: int) -> None:
        """
        draw histograms for each topics distribution
        :param results:
        :param topics_num:
        :return:
        """
        for topic in list(range(0, topics_num)):
            title = "Topic " + str(topic) + " Distribution by Article"
            results[results['max_score_index'] == topic].hist(column='max_score', grid=False)
            plt.title(title)
            plt.ylabel("document_count")
            plt.xlabel("topic_score")
            plt.savefig(title + ".png")


class NewsScrapper:
    def __init__(self):
        self.news_source_list = [
                "https://news.google.com/rss/search?q=ironman&hl=en-US&gl=US&ceid=US:en",
                "https://news.google.com/rss/search?q=ant-man+and+the+wasp&hl=en-US&gl=US&ceid=US:en"
            ]
        self.NONE_STR = "None"
        self.timezone = pytz.timezone("Asia/Tokyo")
        self.NEWS_DATE_FORMAT = "%a, %d %b %Y %H:%M:%S %Z"
        self.local_format = "%Y-%m-%d %H:%M:%S"
        self.timezone = pytz.timezone("Asia/Tokyo")

    def str_to_datetime(self, input_date_str: str, date_format: str = None) -> Optional[datetime]:
        """
        convert date string to datetime with date_format
        default date-format is self.DATE_TIME_FORMAT
        :param input_date_str:
        :param date_format:
        :return:
        """
        if not input_date_str or input_date_str == self.NONE_STR:
            return None
        return self.timezone.localize(datetime.strptime(input_date_str, date_format))

    def get_news_lists(self) -> DataFrame:
        """
        collect the raw data list
        :return:
        """
        data = defaultdict(list)
        for url in self.news_source_list:
            print(url)
            with urlopen(url) as file:
                xml_page = file.read()
            soup_page = BeautifulSoup(xml_page, "xml")
            for news in soup_page.find_all("item"):
                data["title"].append(news.title.text)
                data["description"].append(BeautifulSoup(news.description.text, "xml").text)
                data["link"].append(news.link.text)
                data["source"].append(news.source.text)
                data["pub_date"].append(datetime.strptime(news.pubDate.text, self.NEWS_DATE_FORMAT).replace(tzinfo=pytz.utc).astimezone(
                    self.timezone).strftime(self.local_format))
                print("crawl data from", news.link.text)
                data["news_data"].append(self.get_text_from_link(news.link.text))

        return DataFrame(data)

    @staticmethod
    def get_text_from_link(news_url: str) -> str:
        """
        collect text from news_url
        :param news_url:
        :return:
        """
        def remove_unused_text(soup: BeautifulSoup) -> BeautifulSoup:
            """
            clean soup data
            :param soup:
            :return:
            """
            for item in soup:
                if isinstance(item, Comment):
                    item.extract()
            for remove_tag in soup(["footer", "nav", "header", "code", "table", "channels-list", "disclaimer",
                                    "router-outlet", "article-header", "img", "button", "sub-channels-list",
                                    "progressive-image", "font", "br"]):
                remove_tag.decompose()

            for script in soup(["script", "style", "noscript"]):
                script.extract()
            return soup

        try:
            page = requests.get(news_url)
            if page.status_code != 200:
                # if can't crawl return empty string
                return ""
            soup = BeautifulSoup(page.content, 'html.parser')
            soup = remove_unused_text(soup)
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            # remove punctuation
            chunks = (re.sub(r'[^\w\s]', '', phrase.strip()) for line in lines for phrase in line.split("  "))
            # drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk and len(chunk.split()) > 5)
            return text
        except:
            print("There is error  %s", traceback.format_exc())
            return ""

    @staticmethod
    def save_news_to_csv(news_data: DataFrame, topic_num):
        """
        :param news_data:
        :param topic_num:
        :return:
        """
        # save all results
        news_data.to_csv("results.csv", index=False, encoding="utf-8")

        # save all result by topic with crawl news data
        for topic in list(range(0, topic_num)):
            news_data[news_data["max_score_index"] == topic].nlargest(20, 'max_score').to_csv("results_topic_" + str(topic) + ".csv", index=False,
                                                                                      encoding="utf-8")

        # save top 20 by topic
        for topic in list(range(0, topic_num)):
            news_data[news_data["max_score_index"] == topic].nlargest(20, "max_score").to_csv(
                "top_20_for_topic_" + str(topic) + ".csv", index=False,
                encoding="utf-8")

        # save records without crawl news data
        for topic in list(range(0, topic_num)):
            with open("results_topic_without_news_" + str(topic) + ".csv", mode="w", newline="\n",
                      encoding="utf-8") as employee_file:
                employee_writer = csv.writer(employee_file, delimiter=',', quoting=csv.QUOTE_ALL)
                employee_writer.writerow(["title", "description", "link", "source", "pub_date", "max_score", "topic"])
                for index, row in news_data[news_data["max_score_index"] == topic].nlargest(20, "max_score").iterrows():
                    employee_writer.writerow(
                        [row["title"], row["description"], row["link"], row["source"], row["max_score"],
                         row["max_score_index"]])

    @staticmethod
    def save_with_diff_cat(new_data: DataFrame, num_topics):
        """
        unused function remove later
        find sub topics for each topics
        :param new_data:
        :param num_topics:
        :return:
        """
        #
        topic_by_cat = defaultdict(DataFrame)
        for topic in list(range(0, num_topics)):
            topic_by_cat[topic] = classifier.get_max_topic_score(new_data[(new_data["max_score_index"] == topic) &
                                                                        (new_data["max_score"] >= 0.9)],
                                                                 number_topics, "topic_by_cat" + str(topic))

        for key, value in topic_by_cat.items():
            for topic in list(range(0, num_topics)):
                with open("topic_by_cat" + str(key) + str(topic) + ".csv", mode="w", newline="\n",
                          encoding="utf-8") as file:
                    writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_ALL)
                    writer.writerow(["Title", "Description", "link", "source", "pub_date", "max_score", "topic"])
                    for index, row in value[value["max_score_index"] == topic].nlargest(20, 'max_score').iterrows():
                        writer.writerow(
                            [row["title"], row["description"], row["link"], row["source"], row["max_score"],
                             row["max_score_index"]])


if __name__ == "__main__":
    news_collector = NewsScrapper()
    classifier = NewsDataClassifier()
    number_topics = 4
    # data from scrapper
    clean_news = classifier.clean_data(news_collector.get_news_lists())
    clean_news = clean_news.drop_duplicates()

    # divided the group again
    # news_collector.save_with_diff_cat(clean_news, number_topics)
    result = classifier.get_max_topic_score(clean_news, number_topics, "by_topic")

    # save result to csv
    news_collector.save_news_to_csv(result, number_topics)

    # save histogram for each topics
    classifier.get_histograms(result, number_topics)
