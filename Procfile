web: gunicorn -w 2 -b 0.0.0.0:$PORT api.app:app
fastapi: uvicorn api.fastapi_app:app --host 0.0.0.0 --port $PORT
streamlit: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
