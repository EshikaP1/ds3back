from contextlib import nullcontext
from flask import Blueprint, jsonify
from flask_restful import Api, Resource
import json
import random
from flask import request

recipe_api = Blueprint('recipe_api', __name__, url_prefix='/api/recipe')
api = Api(recipe_api)

def beautify_json_data(json_file_path):
    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        recipes = []
        for item in data.get('items', []):
            beautified_item = {
                "id": item.get("id", 0),
                "recipe": item.get("recipe", ""),
                "ingredients": item.get("ingredients", ""),
                "time": item.get("time", ""),
                "type": item.get("genre", ""),
            }

            recipes.append(beautified_item)

        return recipes

    except FileNotFoundError:
        return {"error": "File not found"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format in the file"}

class _Read(Resource):
    def get(self):
        json_list = beautify_json_data('model/recipe.json')
        return jsonify(json_list)

class _ReadRandom(Resource):
    def get(self):
        recipes = beautify_json_data('model/recipe.json')
        random_item = random.choice(recipes)
        return jsonify(random_item)

class _Search(Resource):
    def get(self):
        query = request.args.get('query')
        if not query:
            return {"error": "No query provided"}, 400

        recipes = beautify_json_data('model/recipe.json')
        results = [item for item in recipes if query.lower() in item['recipe'].lower()]

        return jsonify(results)

class _SearchByGenre(Resource):
    def get(self):
        genre = request.args.get('genre')
        if not genre:
            return {"error": "No genre provided"}, 400

        recipes = beautify_json_data('model/recipe.json')
        results = [item for item in recipes if item['type'].lower() == genre.lower()]

        return jsonify(results)

class _Count(Resource):
    def get(self):
        recipes = beautify_json_data('model/recipe.json')
        count = len(recipes)
        return {"count": count}

api.add_resource(_Read, '/')
api.add_resource(_ReadRandom, '/random')
api.add_resource(_Search, '/search')
api.add_resource(_SearchByGenre, '/searchbygenre')
api.add_resource(_Count, '/count')