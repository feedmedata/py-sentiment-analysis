from sys import argv

from pandas import read_csv
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
plt.style.use('ggplot')

POSITIVE = 1.0
NEUTRAL  = 0
NEGATIVE = -1.0


def feedback_bar(df: DataFrame, arg: str) -> bool:
    """
    Displays or saves bar chart for feedback type count.
    :param df:DataFrame
    :arg:String - argument flag indicating chart display
    :return bool
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

    if arg == '-s':
        plt.savefig('feedback_bar.png')
        return True
    elif arg == '-d':
        plt.show()
        return False


def review_histogram(df: DataFrame, arg: str) -> bool:
    """
    Displays or saves histogram for description lengths.
    :param df:
    :arg:String - argument flag indicating chart display
    :return bool
    """
    df.hist(column='length', grid=True, bins=8000, cumulative=False)
    plt.title('Histogram of descriptions length')
    plt.axis([20, 500, 0, 15000])
    plt.ylabel('Count')
    plt.xlabel('Length')

    if arg == '-s':
        plt.savefig('length_histogram.png')
        return True
    elif arg == '-d':
        plt.show()
        return False

def filter_df(df: DataFrame, length_gte: int = None, rates : int = None) -> DataFrame:
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


def no_correlation(df: DataFrame) -> DataFrame:
    """
    Counts 'nie' word occurrence in descriptions.
    Adds result to the no_count column and returns
    correlation matrix.
    :param df:DataFrame
    :return:DataFrame correlation matrix
    """
    df['no_count'] = 0

    for i, row in df.iterrows():
        no_count = 0
        desc_list = row.description.split(' ')
        for word in desc_list:
            if word.lower() == 'nie':
                no_count += 1
        df.at[i, 'no_count'] = no_count
    return df.corr(method='pearson')


if __name__ == '__main__':
    if argv[1] not in ['-s', '-d']:
        raise ValueError('Please, provide proper argument according to the documentation.')

    #Reading data from .csv and putting it into pandas DataFrame.
    df = read_csv(argv[2])
    #Cleaning DataFrame, dropping rows with NaN
    df = df.dropna()

    #Task 1) Feedback visualization
    print("1) Feedback visualization:")
    if feedback_bar(df, argv[1]):
        print('Feedback chart saved as feedback_bar.png')
    else:
        print('Feedback chart displayed.')

    #Task 2) Histogram for description lengths
    print("\n2) Length histogram")
    if review_histogram(df, argv[1]):
        print('Length histogram saved as length_histogram.png')
    else:
        print("Histogram displayed.")

    #Task 3) Positive and negative feedback filtering
    filtered_df = filter_df(df, 100, [POSITIVE, NEGATIVE])
    no_neutral = NEUTRAL not in filtered_df.rate
    len_gte_100 = min(filtered_df.length) > 99
    assert no_neutral
    assert len_gte_100
    print(f"\n3) Filtering DataFrame (length > 100 and rate in [POSITIVE, NEGATIVE]")
    print(f"Neutral not in df.rate: {no_neutral}")
    print(f"Minimal length of description > 99: {len_gte_100}")

    #Task 4) Finding correlation between length and rate.
    corr = df.corr(method='pearson')
    print(f"\n4) Length and rate correlation\n {corr}")

    #Task 5) Interesting relation between features ('Nie' word usage to rate type).
    no_corr = no_correlation(df)
    print(f"\n5) Number of 'nie' word used and rate correlation\n {no_corr}")
