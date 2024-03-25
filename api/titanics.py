import json, jwt
from flask import Blueprint, request, jsonify, current_app, Response
from flask_restful import Api, Resource # used for REST API building
from datetime import datetime
from model.titanic import predictSurvival
import numpy as np
from flask_cors import CORS
from flask import Flask
titanic_api = Blueprint('titanic_api', __name__,
                   url_prefix='/api/titanic')

# API docs https://flask-restful.readthedocs.io/en/latest/api.html
api = Api(titanic_api)
app = Flask(__name__)
CORS(app)
class Titanic:        
    class _titanic(Resource):
        def post(self): # Create method
            ''' Read data for json body '''
            body = request.get_json()
            print(body)
            response = predictSurvival(body)
            print(response)
            return jsonify(int(response))
            
            
    api.add_resource(_titanic, '/')
            