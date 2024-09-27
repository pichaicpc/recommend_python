from flask import Flask,send_file, abort
from sklearn.neighbors import NearestNeighbors
import mysql.connector as sql
import pandas as pd
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Create the Flask app (API)
app = Flask(__name__)

# Connect to the database
conn = sql.connect(host="localhost", 
                   database="shopdee", 
                   user="root", 
                   password="1234")


# Define the API endpoint
@app.route('/api/recommend/<int:id>', methods=['GET'])
def recommend(id):

    #Load data
    sql = "CALL GetProductCountsAndIDs()"
    x = pd.read_sql(sql, conn, index_col=['custID'])

    #Define prediction data (test data in production) and traning data
    x_login_user = x[x.index==id] #prediction data
    x_other_users = x.drop(x[x.index==id].index) #traning data

    #Build model 
    net = NearestNeighbors(n_neighbors=1, metric='minkowski', p=2)
    net.fit(x_other_users)

    #Use model (ใช้โมเดลเพื่อหาลูกค้าอื่นที่มีพฤติกรรมคล้ายกับลูกค้าที่ Login เข้ามามากที่สุด)
    index = net.kneighbors(x_login_user)[1][0][0]

    #ได้ลูกค้าที่มีพฤติกรรมคล้ายกันมากที่สุด
    x_similar = x_other_users.iloc[[index]]

    #หารหัสสินค้าที่จะแนะนำ
    #โดยพิจารณาจากสินค้าที่ลูกค้าที่ login เข้ามาไม่เคยซื้อ แต่ลูกค้าที่มีพฤติกรรมคล้ายกันเคยซื้อ
    product_count = len(x_login_user.columns)
    product_ids = []

    for i in range(product_count):    
        if(x_login_user.iloc[0,i] == 0 and x_similar.iloc[0,i] > 0):
            product_ids.append(int(x_similar.columns[i]))

    product_ids = ', '.join(map(str, product_ids))

    if len(product_ids) == 0:
        return [], 200


    #แสดงรายการสินค้าที่ต้องการแนะนำ
    sql = f'SELECT * FROM product WHERE productID IN ({product_ids})'
    if not conn.is_connected():
        conn.reconnect()
    products = pd.read_sql(sql, conn)    

    # Return the result
    return products.to_dict(orient='records'), 200


@app.route("/api/product/image/<filename>")
def get_product_image(filename):

    # Directory containing product images
    IMAGE_DIR = os.path.join(os.path.dirname(__file__), 'assets/product')

    # Construct the file path
    filepath = os.path.join(IMAGE_DIR, filename)

    # Check if the file exists
    if not os.path.exists(filepath):
        abort(404, description="Image not found")

    # Return the image file
    return send_file(filepath)


# Create Web server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)