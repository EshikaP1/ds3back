import json, jwt
from flask import Blueprint, request, jsonify, current_app, Response
from flask_restful import Api, Resource # used for REST API building
from datetime import datetime
from auth_middleware import token_required

from model.exercise import ExerciseRegression

exercise_api = Blueprint('exercise_api', __name__,
                   url_prefix='/api/exercise')

# API docs https://flask-restful.readthedocs.io/en/latest/api.html
api = Api(exercise_api)

class ExcerciseAPI:        
    class _predict(Resource):  # User API operation for Create, Read.  THe Update, Delete methods need to be implemeented
        def post(self): # Create method
            ''' Read data for json body '''
            contestant = request.get_json()
            response = ExerciseRegression.predictWeight(contestant)
            return jsonify(response)
            
    api.add_resource(_predict, '/')


# class _predict(Resource):
        # Define the API endpoint for prediction
        #@app.route('/api/predict', methods=['POST'])
        #def predict():
            # Get the passenger data from the request
            #passenger = request.get_json()

            #response = predictSurvival(passenger)

            # Return the response as JSON
            #return jsonify(response)
        
