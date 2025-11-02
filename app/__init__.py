import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
from flask_wtf.csrf import CSRFProtect
from datetime import timedelta
from flask_session import Session

# Initialize extensions
db = SQLAlchemy()
csrf = CSRFProtect()
bootstrap = Bootstrap()
login_manager = LoginManager()
login_manager.login_view = 'authentication.do_the_login'
login_manager.session_protection = 'strong'
bcrypt = Bcrypt()

def create_app(config_type):
    app = Flask(__name__)

    # Load configuration
    configuration_path = os.path.join(os.getcwd(), 'config', f'{config_type}.py')
    app.config.from_pyfile(configuration_path)

    # Update session configuration
    app.config['SESSION_COOKIE_DOMAIN'] = None # Remove domain setting
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=14)
    app.config['SESSION_COOKIE_SECURE'] = False  # Use False for development, True for production with HTTPS
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_FILE_DIR'] = '/Users/shardendujha/thesis-project-final-data/flask_session'  # Ensure this path is correct
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

    # Initialize extensions
    db.init_app(app)
    bootstrap.init_app(app)
    login_manager.init_app(app)
    bcrypt.init_app(app)
    Session(app)

    # Register blueprints
    from app.Indoor_air_quality import main as indoor_air_quality
    from app.auth import authentication as auth

    app.register_blueprint(indoor_air_quality)
    app.register_blueprint(auth)

    return app
