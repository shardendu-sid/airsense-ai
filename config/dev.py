import os

host = os.environ.get('DB_HOST')
port = os.environ.get('DB_PORT')
database = os.environ.get('DB_NAME')
user = os.environ.get('DB_USER')
password = os.environ.get('DB_PASSWORD')



DEBUG = True

SECRET_KEY = os.urandom(24)

# SQLALCHEMY_DATABASE_URI = 'postgresql://shardendu:computer@localhost:5432/awair'

SQLALCHEMY_DATABASE_URI = f"postgresql://{user}:{password}@{host}:{port}/{database}"
SQLALCHEMY_TRACK_MODIFICATIONS = False