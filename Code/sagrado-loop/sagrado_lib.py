import pymysql.cursors
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from a .env file (containing OPENAI_API_KEY)
load_dotenv()


# Function to establish a MySQL connection
def create_mysql_connection():
    host = os.environ.get('MYSQL_HOST')
    user = os.environ.get('MYSQL_USER')
    password = os.environ.get('MYSQL_PASSWORD')
    database = os.environ.get('MYSQL_DATABASE')
    # print(host, user, password, database)
    connection = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        cursorclass=pymysql.cursors.DictCursor
    )
    # print(connection)
    return connection


# Function to get the latest values from the MySQL database
def latest_values():
    # Establish a connection to the MySQL server
    connection = create_mysql_connection()
    # Create a cursor object to execute SQL queries
    cursor = connection.cursor()
    # Specify the table name and achor collum in the SQL query
    query = "SELECT * FROM sbwpcb ORDER BY time DESC LIMIT 1"
    # Execute the SQL query
    cursor.execute(query)
    # connection.commit()
    # Fetch the results
    last_row = cursor.fetchone()

    #Convert to list
    last_row = list(last_row.values())

    # Close the cursor and connection
    cursor.close()
    connection.close()

    # Create a dictionary with the values
    latest_data = {
        "time": last_row[0],
        "biodata1": last_row[1],
        "biodata2": last_row[2],
        "pulsesensor": last_row[3],
        "moisture": last_row[4],
        "temperature": last_row[5],
        "humidity": last_row[6],
        "pressure": last_row[7],
        "gas": last_row[8]
    }
    return latest_data


#Function to create a dataframe from mysql data
def sql_dataframe():
    # Establish a connection to the MySQL server
    connection = create_mysql_connection()
    cursor = connection.cursor()
    # Create query with table name
    query = "SELECT * FROM sbwpcb"
    cursor.execute(query)
    data = cursor.fetchall()

    # Close the connection
    connection.close()

    # Return dataframe
    # print('Data: ', data)
    return data

# Initial dataframe request
df = sql_dataframe()
print('Data: ', data)


# Initialize the PandasAI object
openai_api_key = os.environ.get('OPENAI_API_KEY')