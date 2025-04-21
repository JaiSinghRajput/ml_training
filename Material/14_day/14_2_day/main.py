from flask import Flask, render_template
import mysql.connector

# Establish a connection to the MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Rajput@9461",
    database="price_prediction"
)

# Create a cursor object to interact with the database
cursor = db.cursor()

# Example query to create a table
cursor.execute("""
CREATE TABLE IF NOT EXISTS demo_table (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255),
    age INT
)
""")

# Example query to insert data into the table
cursor.execute("INSERT INTO demo_table (name, age) VALUES (%s, %s)", ("John Doe", 30))
db.commit()

# Example query to fetch data from the table
cursor.execute("SELECT * FROM demo_table")
rows = cursor.fetchall()
for row in rows:
    print(row)

# Close the cursor and database connection
cursor.close()
db.close()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/history1')
def history1():
    return render_template('history1.html')

@app.route('/project')
def project():
    return render_template('project.html')

if __name__ == '__main__':
    app.run(debug=True)