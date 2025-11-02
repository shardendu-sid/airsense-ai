#  create blueprint 

from flask import Blueprint

main = Blueprint('main', __name__, template_folder='templates')

from app.Indoor_air_quality import routes