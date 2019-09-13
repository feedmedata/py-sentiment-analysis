from sys import argv

from pandas import read_csv
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
plt.style.use('ggplot')

CSV = ''

POSITIVE = 1.0
NEUTRAL = 0
NEGATIVE = -1.0

def save_or_display(func):
    def handle_output():
        func()
        if argv[2] == '-s':
            plt.savefig('feedback_bar.png')
        else:
            plt.show()
    return handle_output()


#@save_or_display
def feedback_bar(df):
    """
    Displays or saves bar chart for feedback type count.
    :param df:DataFrame
    """
    labels = ['Positive', 'Neutral', 'Negative']
    counts = [len(df[df.rate == POSITIVE]),
              len(df[df.rate == NEUTRAL]),
              len(df[df.rate == NEGATIVE])]

    plt.figure(figsize=(12, 8))
    plt.bar(labels, counts, width=0.4)
    plt.title('Feedback type count')

    ax = plt.gca()

    ax.set_xlabel('Feedback type', fontsize=12)
    ax.xaxis.set_label_coords(0.5, -0.1)

    ax.set_ylabel('Number of feedbacks', fontsize=12)
    ax.yaxis.set_label_coords(-0.12, 0.5)

    for i in range(3):
        plt.text(x=i-0.06, y=counts[i], size=10, s=counts[i])

    #plt.savefig('feedback_bar.png')


#@save_or_display
def review_histogram(df):
    """
    Displays or saves histogram for description lengths.
    :param df:
    """
    df['desc_len'] = df.apply(lambda row: len(row.description), axis=1)
    df.hist(column='desc_len')
    plt.savefig('histogram.png')
    #TODO!


def filter(df, length_gte=None, rates=None) -> DataFrame:
    """
    Returns filtered DataFrame descriptions.
    Possible filters:
        1) description text length greater than or equal :length_gte
        2) description rate in :rates

    :param df:DataFrame
    :param length_gte:int
    :param rates:[POSITIVE, NEUTRAL, NEGATIVE]
    :return:DataFrame
    """
    if length_gte:
        df = df[df.length >= length_gte]
    if rates:
        df = df[df.rate.isin(rates)]
    return df


def creative_function():
    ...


if __name__ == '__main__':
    #Reading data from .csv and putting it into pandas DataFrame.
    df = read_csv(CSV)
    #Cleaning DataFrame, dropping rows with NaN
    df = df.dropna()

    #Task 1) Feedback visualization
    feedback_bar(df)

    #Task 2) Histogram for description lengths
    #review_histogram(df)

    #Task 3) Positive and negative feedback filtering
    filtered_df = filter(df, 100, [POSITIVE, NEGATIVE])
    assert NEUTRAL not in filtered_df.rate
    assert min(filtered_df.length) > 99

    #Task 4) Finding correlation between length and rate.
    corr = df.corr(method='pearson')
    print(f"Length and rate correlation\n\n {corr}")

    #Task 5) Interesting relation between features.
