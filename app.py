from flask import Flask, jsonify, request, send_file
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask_cors import CORS
from flask_cors import cross_origin
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import io
import psycopg2
import numpy as np
import kaleido  # Ensure kaleido is installed
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime  # Import datetime module
import base64
from dotenv import load_dotenv
import os
from urllib.parse import urlparse

# Load environment variables from the .env file
load_dotenv(dotenv_path='../.env')
app_port = os.getenv('APP_PORT')
print(app_port)

app = Flask(__name__)
CORS(app)

analyzer = SentimentIntensityAnalyzer()

# Determine if DATABASE_URL is provided (e.g., in a production environment)
DATABASE_URL = os.getenv('DATABASE_URL')

if DATABASE_URL:
    # Parse the DATABASE_URL for production
    url = urlparse(DATABASE_URL)
    db_config = {
        'dbname': url.path[1:],  # Remove leading '/'
        'user': url.username,
        'password': url.password,
        'host': url.hostname,
        'port': url.port,
        'sslmode': 'require'  # Enforce SSL for production
    }
else:
    # Default to environment variables for local development
    db_config = {
        'dbname': os.getenv('DB_DATABASE', 'lolos_place_database'),
        'user': os.getenv('DB_USER', 'lolos_place_database_user'),
        'password': os.getenv('DB_PASSWORD', 'kxwp1hAcA2psjJr8fNsqQSdWjreTBC5F'),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432')
    }

# Function to get the database connection
def get_db_connection():
    return psycopg2.connect(**db_config)




@app.route('/')
def home():
    return 'Flask app is working!'




from datetime import time  # Import the time class
@app.route('/test-db')
def test_db():
    try:
        # Establish a database connection
        conn = get_db_connection()
        cursor = conn.cursor()

        # Execute a lightweight query to test the connection
        cursor.execute("SELECT 1;")

        # Close the cursor and connection
        cursor.close()
        conn.close()

        return jsonify({'message': 'Database connected successfully flask'})
    except Exception as e:
        return jsonify({'error': 'Failed to connect to the database', 'details': str(e)}), 500




@app.route('/sales-forecast', methods=['GET'])
def sales_forecast():
    try:
        # Connect to the database
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        EXTRACT(YEAR FROM CAST(date AS DATE)) AS year,
                        EXTRACT(MONTH FROM CAST(date AS DATE)) AS month,
                        SUM(gross_sales) AS total_gross_sales
                    FROM sales_data
                    WHERE EXTRACT(YEAR FROM CAST(date AS DATE)) >= 2019
                    GROUP BY year, month
                    ORDER BY year, month;
                """)
                data = cursor.fetchall()

        # If no data is found
        if not data:
            return jsonify({"error": "No sales data available for forecasting"}), 404

        # Prepare data for forecasting
        df = pd.DataFrame(data, columns=['year', 'month', 'total_gross_sales'])

        # Combine year and month to create a date column (first day of each month)
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

        # Ensure the data has valid values
        df = df.dropna(subset=['date', 'total_gross_sales'])

        # Check if the cleaned data is empty
        if df.empty:
            return jsonify({"error": "Data is empty after cleaning"}), 404

        # Convert date to ordinal for modeling
        df['date_ordinal'] = df['date'].apply(lambda x: x.toordinal())

        # Prepare independent (X) and dependent (y) variables
        X = df['date_ordinal'].values.reshape(-1, 1)
        y = df['total_gross_sales'].values

        # Ensure there is enough data for linear regression
        if len(X) < 2:  # Need at least two data points to fit the model
            return jsonify({"error": "Not enough data to fit the model"}), 400

        # Create a Linear Regression model and fit the data
        model = LinearRegression()
        model.fit(X, y)

        # Calculate predicted sales for the current month
        current_month = datetime.now().month
        current_year = datetime.now().year
        current_month_date = datetime(current_year, current_month, 1)
        current_month_ordinal = np.array([current_month_date.toordinal()]).reshape(-1, 1)
        predicted_sales_this_month = model.predict(current_month_ordinal)

        # Prepare the result for predicted sales for the current month
        predicted_sales_current_month = {
            'year': current_year,
            'month': current_month,
            'predicted_sales': predicted_sales_this_month.tolist()[0]
        }

        # Prepare historical sales data grouped by year and month
        sales_per_month = df[['year', 'month', 'total_gross_sales']].to_dict(orient='records')

        # Prepare the response data
        response_data = {
            'sales_per_month': sales_per_month,
            'predicted_sales_current_month': predicted_sales_current_month
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": "Error in forecasting sales: " + str(e)}), 500
    







@app.route('/feedback-graph', methods=['GET'])
def feedback_graph():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT sentiment, COUNT(*) 
                    FROM feedback 
                    GROUP BY sentiment;
                """)
                sentiment_data = cursor.fetchall()

        if not sentiment_data:
            return jsonify({"error": "No feedback data available"}), 404

        # Extract sentiments and counts dynamically
        sentiments = [row[0].capitalize() for row in sentiment_data]
        counts = [row[1] for row in sentiment_data]

        # Define default colors for known sentiments
        sentiment_colors = {
            "Positive": "green",
            "Negative": "red",
            "Neutral": "gray"
        }
        # Assign colors dynamically, default to blue for unknown sentiments
        colors = [sentiment_colors.get(sentiment, "blue") for sentiment in sentiments]

        # Generate the pie chart
        fig = go.Figure(data=[go.Pie(labels=sentiments, values=counts, marker=dict(colors=colors))])

        fig.update_layout(
            title="Sentiment Distribution",
            showlegend=True,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="black"),
        )

        # Convert the figure to an SVG image using Kaleido
        img_io = io.BytesIO()
        pio.write_image(fig, img_io, format='svg')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/svg+xml')

    except Exception as e:
        print("Error generating feedback graph:", e)
        return jsonify({"error": "Error generating graph"}), 500













@app.route('/feedback-stats', methods=['GET'])
def feedback_stats():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT sentiment, COUNT(*) 
                    FROM feedback 
                    GROUP BY sentiment;
                """)
                sentiment_data = cursor.fetchall()

        if not sentiment_data:
            return jsonify({"error": "No feedback data available"}), 404

        # Prepare the data for response
        total_feedbacks = sum(row[1] for row in sentiment_data)
        feedback_stats = {
            "total": total_feedbacks,
            "positive": next((row[1] for row in sentiment_data if row[0].lower() == "positive"), 0),
            "negative": next((row[1] for row in sentiment_data if row[0].lower() == "negative"), 0),
            "neutral": next((row[1] for row in sentiment_data if row[0].lower() == "neutral"), 0),
        }

        return jsonify(feedback_stats)

    except Exception as e:
        print("Error fetching feedback stats:", e)
        return jsonify({"error": "Error fetching feedback stats"}), 500








# Negative words that could appear in text
negative_words = ['bad', 'sad', 'angry', 'hate', 'worst', 'terrible', 'awful', 'dislike', 'sinful']

# Specific cases where negative words indicate positive sentiment
positive_with_negative_words = [
    "sinful",  # This could be used in a positive way when describing indulgence, like in desserts
    "bad"  # Sometimes 'bad' is used in a playful or indulgent context (e.g., "This is bad, but so good")
]




@app.route('/api/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()  # Getting the JSON data from the request
    text = data.get('text')  # Extracting the text field from the JSON data
    
    # Analyzing sentiment using the VADER sentiment analyzer
    sentiment_score = analyzer.polarity_scores(text)
    compound_score = sentiment_score['compound']
    
    # Check for the presence of negative words, but allowing for positive sentiment context
    contains_negative_word = any(neg_word in text.lower() for neg_word in negative_words)
    
    # Special case for identifying positive sentiment with negative words
    sentiment_label = ''
    
    # Check if the text contains any words that should be considered as positive in context
    if contains_negative_word:
        if any(phrase in text.lower() for phrase in positive_with_negative_words):
            sentiment_label = 'positive sentiment with negative words'
        elif compound_score > 0.5:  # If VADER analysis indicates overall positive sentiment
            sentiment_label = 'positive sentiment with negative words'
        elif compound_score < -0.5:  # If the sentiment score is clearly negative
            sentiment_label = 'negative sentiment with negative words'
        else:
            sentiment_label = 'neutral sentiment with negative words'
    else:
        # If no negative words are detected, proceed with regular sentiment analysis
        if compound_score > 0.5:
            sentiment_label = 'positive'
        elif compound_score < -0.5:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'

    # Returning the sentiment result as a JSON response
    return jsonify({
        'compound': compound_score,
        'sentiment': sentiment_label
    })



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app_port)
