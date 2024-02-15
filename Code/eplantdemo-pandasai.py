"""
This Python script powers a Streamlit web application designed for an innovative interaction with a plant.
The app integrates with a MySQL database to retrieve real-time data about the plant's physiological and
environmental conditions. It utilizes the PandasAI library, which in turn employs OpenAI's language models,
to enable users to 'chat' with the plant. The script fetches and displays various metrics such as root and
leaf conductivity, humidity, temperature, and other environmental factors. It offers a unique user
experience by allowing queries to be posed to the plant, which are then answered based on the collected
data, simulating a conversation. The application also includes visualizations of the data using Matplotlib
and manages environment variables for secure API and database access.
"""


import streamlit as st
import pandas as pd
import mysql.connector
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt
from streamlit_chat import message
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
    connection = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
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
    # Fetch the results
    last_row = cursor.fetchone()
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
    # Create query with table name
    query = "SELECT * FROM sbwpcb"
    # Use pandas.read_sql() to execute the query and retrieve data into a DataFrame
    df = pd.read_sql(query, con=connection)
    # Close the connection
    connection.close()
    # Return dataframe
    return df

# Initial dataframe request
df = sql_dataframe

# Initialize the PandasAI object
openai_api_key = os.environ.get('OPENAI_API_KEY')
llm = OpenAI(openai_api_key)
pandas_ai = PandasAI(llm, conversational=True)

# Streamlit app configuration
st.set_page_config(page_title="1st E-plant Chatroom", page_icon="🌱")

# Define the prompt context
prompt_context = f"Hi model. From now on, you are a plant. You will not responde as an AI languate model. You'll always roleplay a plant whose pyshiological and environmental conditions are being recorded in the dataset {df}"

# Image
image = "/Users/danilovaz/Desktop/sandbox/ecanaris/ecanaris.jpeg"
caption = "Setup"

# Streamlit app
# Streamlit app
def main():
    # Get the latest values
    latest_data = latest_values()
    
    #App
    st.title("E-plant Chat")
    st.image(image, caption=caption)
    st.markdown("With this interface you can talk to a plant connected to the internet, accessing real-time data about its roots' electric activity & its environmental conditions. 'Roots Conductivity' is using a 12-bit Analog-to-Digital Converter (ADC) circuit connected to the plant's roots (with a 10k pullup resistor), in a scale that goes from 0 to 4095. 'Roots COnductivity 2' reflects the same reading, but in milivoltage. The plant's surrounding environment is being monitored by an Adafruit BME680 - Temperature, Humidity, Pressure and Gas Sensor, that measures 'humidity' with ±3% accuracy, barometric 'pressure' with ±1 hPa absolute accuracy, 'temperature' with ±1.0°C accuracy, and 'gas' is a values based on the volatile organic compounds (VOC) in the air.")
    st.subheader("Latest Readings")
    col1, col2, col3 = st.columns(3)
    col1.metric("TimeStamp", latest_data['time'])
    col2.metric("Humidity", latest_data['humidity'])
    col3.metric("Temperature", latest_data['temperature'])
    col4, col5, col6 = st.columns(3)
    col4.metric("Roots Conductivity", latest_data['biodata1'])
    col5.metric("Leaves Conductivity", latest_data['biodata2'])
    col6.metric("steam PPG", latest_data['pulsesensor'])
    col7, col8, col9 = st.columns(3)
    col4.metric("Gas", latest_data['gas'])
    col5.metric("Pressure", latest_data['pressure'])
    col6.metric("Moisture", latest_data['moisture'])

    # User input
    question = st.text_input("Ask the plant a question:")

    # Process user input and get response from the plant
    if question:
        latest_data = latest_values()
        response = pandas_ai(df, prompt=prompt_context + " " + question)

        fig = None  # Initialize fig as None

        # Check if there are figures to display
        if len(plt.get_fignums()) > 0:
            fig = plt.gcf()

        # Store the conversation history and plots as tuples
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        st.session_state['chat_history'].append((question, response, fig))

        # Display the conversation history and plots
        for i, (q, r, f) in enumerate(st.session_state['chat_history']):
            message(q, is_user=True, key=str(i) + '_user')
            message(r, key=str(i))
            if f is not None:
                st.pyplot(f)


if __name__ == "__main__":
    main()

