import sys
import logging
import re
import pandas as pd
from sqlalchemy import create_engine


#set logging configuration
logging.basicConfig(encoding='utf-8',
                    format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def load_data(messages_filepath, categories_filepath):
    """Extract data form csv files and merge them into dataframe

    :param messages_filepath: path to csv file containing message data
    :param categories_filepath: path to csv file containing category data
    :return: merge dataframe
    """

    logging.info("run load_data")

    # loading of message data
    messages_df = pd.read_csv(messages_filepath)
    logging.info(f"message dataframe frame loaded: {messages_df.shape}")

    # loading of categories data
    categories_df = pd.read_csv(categories_filepath)
    logging.info(f"category dataframe loaded: {categories_df.shape}")

    # merge both dataframes
    merged_df = messages_df.merge(categories_df, on=['id'])
    logging.info(f"category and message are merged: {merged_df.shape}")

    return merged_df


def clean_data(df):
    """Transform clean-up loaded dataframe with both message and category data

    :return: clean dataframe
    """

    logging.info("run clean_data")

    # the 36 categories currently stored in a single string
    # split categories into 36 new columns
    categories_df = df['categories'].str.split(";", expand=True)
    logging.info(f"categories string splitted up in temp dataframe: {categories_df.shape}")

    # category names currently with values. Extract them and rename columns accordingly.
    categories_df.columns = categories_df.iloc[0].apply(lambda x: x[:-2])

    # remove category names attached to all values
    categories_df = categories_df.applymap(lambda x: re.findall(r'(\d)', x)[0])

    # convert all values to numeric type
    categories_df = categories_df.apply(pd.to_numeric)

    # remove former category column and concat new clean caregories data to df
    df = df.drop("categories", axis=1)
    df = pd.concat([df, categories_df], axis=1)
    logging.info("categories data converted to numeric type")

    #  remove duplicates
    df = df.drop_duplicates()
    logging.info("duplicated entry rows are removed")

    return df


def save_data(df, database_filename):
    """Load clean transformed data into database

    :param database_filename: path to database file
    """

    logging.info("run save_data")

    # create engine and connect to file based-database
    engine = create_engine(f"sqlite:///{database_filename}")

    #save data into db file
    df.to_sql('data', engine, if_exists='replace', index=False)
    logging.info(f"data dumped in db file: {database_filename}")


def main():
    """load dataframes form csv files and merge them into dataframe

    :param messages_filepath: path to csv file containing message data
    :param categories_filepath: path to csv file containing category data
    :return: merge dataframe
    """

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        logging.info("Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}"
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        logging.info("Cleaning data...\n")
        df = clean_data(df)

        logging.info("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)
        
        logging.info("Cleaned data saved to database!")
    
    else:
        logging.info("Please provide the filepaths of the messages and categories "\
              "datasets as the first and second argument respectively, as "\
              "well as the filepath of the database to save the cleaned data "\
              "to as the third argument. \n\nExample: python process_data.py "\
              "disaster_messages.csv disaster_categories.csv "\
              "DisasterResponse.db")



if __name__ == '__main__':
    # process_data.py disaster_messages.csv disaster_categories.csv database.db
    main()