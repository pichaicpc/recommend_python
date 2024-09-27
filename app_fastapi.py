from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from sklearn.neighbors import NearestNeighbors
import mysql.connector as sql
import pandas as pd
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Create the FastAPI app
app = FastAPI()

# Connect to the database
conn = sql.connect(
    host="localhost",
    database="shopdee",
    user="root",
    password="1234"
)

# Define the API endpoint
@app.get('/api/recommend/{id}')
async def recommend(id: int):
    try:
        # Load data
        sql_query = "CALL GetProductCountsAndIDs()"
        x = pd.read_sql(sql_query, conn, index_col=['custID'])

        # Define prediction data (test data in production) and training data
        x_login_user = x[x.index == id]  # prediction data
        x_other_users = x.drop(x[x.index == id].index)  # training data

        # Build model
        net = NearestNeighbors(n_neighbors=1, metric='minkowski', p=2)
        net.fit(x_other_users)

        # Use model to find similar users
        index = net.kneighbors(x_login_user)[1][0][0]

        # Get similar user
        x_similar = x_other_users.iloc[[index]]

        # Find product IDs to recommend
        product_count = len(x_login_user.columns)
        product_ids = []

        for i in range(product_count):    
            if (x_login_user.iloc[0, i] == 0 and x_similar.iloc[0, i] > 0):
                product_ids.append(int(x_similar.columns[i]))

        product_ids = ', '.join(map(str, product_ids))

        if len(product_ids) == 0:
            return JSONResponse(content=[])

        # Fetch recommended products
        sql_query = f'SELECT * FROM product WHERE productID IN ({product_ids})'
        if not conn.is_connected():
            conn.reconnect()
        products = pd.read_sql(sql_query, conn)

        # Return the result
        return JSONResponse(content=products.to_dict(orient='records'))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/product/image/{filename}")
async def get_product_image(filename: str):

    # Directory containing product images
    IMAGE_DIR = os.path.join(os.path.dirname(__file__), 'assets/product')

    # Construct the file path
    filepath = os.path.join(IMAGE_DIR, filename)

    # Check if the file is not exist
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")

    # Return the image file
    return FileResponse(filepath)



# Create Web server
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
