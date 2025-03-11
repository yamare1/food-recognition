from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
import torch
import numpy as np
import re

class QueryProcessor:
    """Processes natural language queries about food and nutrition"""
    
    def __init__(self, nutrition_api):
        self.nutrition_api = nutrition_api
        self.intent_model = None
        self.qa_model = None
        self.intent_tokenizer = None
        self.qa_tokenizer = None
        
        # Intent categories
        self.intents = [
            "nutrition_query",       # What's the protein content in this chicken?
            "comparison_query",      # Is this salad healthier than the pizza?
            "portion_query",         # How large is this serving?
            "dietary_restriction",   # Does this contain gluten?
            "general_info",          # What kind of food is this?
            "meal_planning",         # What goes well with this food?
            "cooking_method",        # How was this food prepared?
            "other"                  # Fallback category
        ]
        
    def load_models(self):
        """Load NLP models for intent classification and question answering"""
        # In a real implementation, you would load specific fine-tuned models
        # For simplicity, we'll use general purpose models as placeholders
        
        print("Loading NLP models...")
        
        # Intent classification model
        intent_model_name = "distilbert-base-uncased"
        self.intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_name)
        self.intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_name)
        
        # Question answering model
        qa_model_name = "distilbert-base-cased-distilled-squad"
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
        
        print("Models loaded successfully")
        
    def _classify_intent(self, query):
        """Classify the intent of the user query"""
        # For demonstration purposes, we'll use rule-based classification
        # In a real implementation, you would use the intent_model
        
        query = query.lower()
        
        # Simple rule-based classification
        if any(word in query for word in ["calorie", "protein", "fat", "carb", "sugar", "nutrition"]):
            return "nutrition_query"
        elif any(word in query for word in ["size", "portion", "serving", "large", "small", "weight"]):
            return "portion_query"
        elif any(word in query for word in ["healthier", "better", "worse", "than", "compare"]):
            return "comparison_query"
        elif any(word in query for word in ["gluten", "dairy", "vegan", "vegetarian", "allergen", "nut"]):
            return "dietary_restriction"
        elif any(word in query for word in ["what", "which", "type", "kind", "is it"]):
            return "general_info"
        elif any(word in query for word in ["with", "pair", "meal", "side", "goes with"]):
            return "meal_planning"
        elif any(word in query for word in ["cooked", "baked", "fried", "grilled", "prepared"]):
            return "cooking_method"
        else:
            return "other"
    
    def extract_food_names(self, query):
        """Extract food names from the query"""
        # In a real implementation, you would use named entity recognition
        # For simplicity, we'll use a rule-based approach
        
        # List of common food words to look for in queries
        common_foods = [
            "pizza", "pasta", "rice", "bread", "apple", "banana", "salad",
            "chicken", "beef", "steak", "fish", "salmon", "potato", "vegetable",
            "fruit", "cheese", "yogurt", "milk", "egg", "sandwich", "burger",
            "fries", "soup", "cake", "cookie", "chocolate", "sushi"
        ]
        
        found_foods = []
        query_lower = query.lower()
        
        # Look for common foods in the query
        for food in common_foods:
            if food in query_lower:
                found_foods.append(food)
        
        return found_foods if found_foods else None
    
    def extract_quantity(self, query):
        """Extract quantity information from the query"""
        # Look for numbers followed by units
        quantity_pattern = r'(\d+(?:\.\d+)?)\s*(g|grams|oz|ounces|servings?|pieces?|slices?)'
        matches = re.findall(quantity_pattern, query, re.IGNORECASE)
        
        if matches:
            amount, unit = matches[0]
            amount = float(amount)
            
            # Convert to grams for consistent processing
            if 'oz' in unit or 'ounce' in unit.lower():
                amount *= 28.35  # 1 oz = 28.35g
            
            return amount
        
        return None
    
    def process_query(self, query, detected_food=None, weight_grams=None):
        """
        Process a natural language query about food
        
        Args:
            query: The user's question
            detected_food: Food detected in the image (optional)
            weight_grams: Estimated weight in grams (optional)
            
        Returns:
            response: Dictionary containing the answer and any relevant data
        """
        # Classify the intent
        intent = self._classify_intent(query)
        
        # Extract foods mentioned in the query
        query_foods = self.extract_food_names(query)
        
        # Determine which food to use
        if query_foods:
            # Use food mentioned in query
            food_name = query_foods[0]
        elif detected_food:
            # Fall back to detected food
            food_name = detected_food
        else:
            return {
                'answer': "I'm not sure which food you're asking about. Could you clarify?",
                'confidence': 0.5,
                'intent': intent
            }
        
        # Extract quantity if present in query
        query_quantity = self.extract_quantity(query)
        
        # Determine which weight to use
        if query_quantity:
            # Use quantity from query
            weight = query_quantity
        elif weight_grams:
            # Fall back to detected weight
            weight = weight_grams
        else:
            # Use default serving size
            weight = 100  # Default to 100g
        
        # Process based on intent
        if intent == "nutrition_query":
            # Get nutrition data
            nutrition = self.nutrition_api.get_nutrition(food_name, weight)
            
            # Extract specific nutrient if asked
            if "protein" in query.lower():
                return {
                    'answer': f"The {food_name} ({weight:.0f}g) contains {nutrition['protein_g']}g of protein.",
                    'nutrition_data': nutrition,
                    'confidence': 0.9,
                    'intent': intent
                }
            elif "fat" in query.lower():
                return {
                    'answer': f"The {food_name} ({weight:.0f}g) contains {nutrition['fat_g']}g of fat.",
                    'nutrition_data': nutrition,
                    'confidence': 0.9,
                    'intent': intent
                }
            elif "carb" in query.lower():
                return {
                    'answer': f"The {food_name} ({weight:.0f}g) contains {nutrition['carbs_g']}g of carbohydrates.",
                    'nutrition_data': nutrition,
                    'confidence': 0.9,
                    'intent': intent
                }
            elif "calorie" in query.lower():
                return {
                    'answer': f"The {food_name} ({weight:.0f}g) contains {nutrition['calories']} calories.",
                    'nutrition_data': nutrition,
                    'confidence': 0.9,
                    'intent': intent
                }
            else:
                # General nutrition info
                return {
                    'answer': f"The {food_name} ({weight:.0f}g) contains {nutrition['calories']} calories, "
                             f"{nutrition['protein_g']}g protein, {nutrition['fat_g']}g fat, and {nutrition['carbs_g']}g carbs.",
                    'nutrition_data': nutrition,
                    'confidence': 0.9,
                    'intent': intent
                }
                
        elif intent == "portion_query":
            nutrition = self.nutrition_api.get_nutrition(food_name, weight)
            return {
                'answer': f"This serving of {food_name} weighs approximately {weight:.0f}g, "
                         f"which is about {nutrition['serving_equivalent']} standard servings.",
                'nutrition_data': nutrition,
                'confidence': 0.8,
                'intent': intent
            }
            
        elif intent == "comparison_query":
            # Extract second food for comparison
            # For simplicity, assume the second food mentioned is the one to compare with
            if len(query_foods) >= 2:
                food2 = query_foods[1]
                nutrition1 = self.nutrition_api.get_nutrition(food_name, 100)  # Standardize to 100g
                nutrition2 = self.nutrition_api.get_nutrition(food2, 100)
                
                if nutrition1.get('status') != 'error' and nutrition2.get('status') != 'error':
                    cal_diff = nutrition1['calories'] - nutrition2['calories']
                    more_or_less = "more" if cal_diff > 0 else "fewer"
                    
                    return {
                        'answer': f"For the same weight (100g), {food_name} has {abs(cal_diff):.0f} {more_or_less} "
                                 f"calories than {food2}. {food_name} has {nutrition1['calories']:.0f} calories, "
                                 f"while {food2} has {nutrition2['calories']:.0f} calories.",
                        'comparison_data': {
                            'food1': {'name': food_name, 'nutrition': nutrition1},
                            'food2': {'name': food2, 'nutrition': nutrition2}
                        },
                        'confidence': 0.8,
                        'intent': intent
                    }
            
            return {
                'answer': "I couldn't find enough information to compare foods. Please specify which foods you want to compare.",
                'confidence': 0.5,
                'intent': intent
            }
            
        elif intent == "dietary_restriction":
            # For simplicity, provide some generic answers based on common foods
            # In a real implementation, you would look up detailed ingredient information
            
            gluten_foods = ["bread", "pasta", "cake", "cookie", "pizza"]
            dairy_foods = ["milk", "cheese", "yogurt", "ice cream"]
            vegan_restricted = ["meat", "chicken", "beef", "fish", "egg", "milk", "cheese", "yogurt"]
            
            if "gluten" in query.lower():
                contains_gluten = any(food in food_name.lower() for food in gluten_foods)
                if contains_gluten:
                    return {
                        'answer': f"{food_name.capitalize()} typically contains gluten or is made with gluten-containing ingredients.",
                        'confidence': 0.7,
                        'intent': intent
                    }
                else:
                    return {
                        'answer': f"{food_name.capitalize()} typically doesn't contain gluten, but this can vary by recipe. Always check the specific ingredients for allergens.",
                        'confidence': 0.6,
                        'intent': intent
                    }
            
            elif any(word in query.lower() for word in ["dairy", "lactose"]):
                contains_dairy = any(food in food_name.lower() for food in dairy_foods)
                if contains_dairy:
                    return {
                        'answer': f"{food_name.capitalize()} contains dairy.",
                        'confidence': 0.7,
                        'intent': intent
                    }
                else:
                    return {
                        'answer': f"{food_name.capitalize()} typically doesn't contain dairy, but this can vary by recipe. Always check the specific ingredients for allergens.",
                        'confidence': 0.6,
                        'intent': intent
                    }
            
            elif any(word in query.lower() for word in ["vegan", "vegetarian"]):
                non_vegan = any(food in food_name.lower() for food in vegan_restricted)
                if non_vegan:
                    return {
                        'answer': f"{food_name.capitalize()} is typically not vegan as it contains or is made with animal products.",
                        'confidence': 0.7,
                        'intent': intent
                    }
                else:
                    return {
                        'answer': f"{food_name.capitalize()} is typically suitable for vegans, but recipes may vary. Check for specific ingredients like honey, eggs, or dairy that might be added.",
                        'confidence': 0.6,
                        'intent': intent
                    }
            
            return {
                'answer': f"I don't have detailed ingredient information for {food_name}. Please check the packaging or recipe for specific dietary restrictions.",
                'confidence': 0.5,
                'intent': intent
            }
            
        elif intent == "general_info":
            # Provide general information about the food
            # In a real implementation, you would look up detailed food information
            
            food_categories = {
                "pizza": "Italian dish consisting of a flattened disk of bread dough topped with various ingredients.",
                "pasta": "Italian food made from a dough using flour, water and eggs. Common types include spaghetti, macaroni, and ravioli.",
                "rice": "Cereal grain that is the most widely consumed staple food for a large part of the world's human population.",
                "bread": "Staple food prepared from a dough of flour and water, usually by baking.",
                "apple": "Fruit produced by the apple tree, rich in fiber and vitamin C.",
                "banana": "Elongated, edible fruit produced by several kinds of large herbaceous flowering plants.",
                "salad": "Dish consisting of mixed vegetables, often with a dressing.",
                "chicken": "Domesticated fowl often served as a food source.",
                "beef": "Meat from cattle, typically used for steaks, roasts, and ground meat.",
                "steak": "Slice of meat, typically beef, cut across the muscle fibers.",
                "fish": "Aquatic animal caught or harvested for consumption.",
                "salmon": "Pink-fleshed fish known for being rich in omega-3 fatty acids.",
                "potato": "Starchy, tuberous crop from the perennial nightshade Solanum tuberosum."
            }
            
            if food_name.lower() in food_categories:
                return {
                    'answer': food_categories[food_name.lower()],
                    'confidence': 0.8,
                    'intent': intent
                }
            else:
                # Generic response
                return {
                    'answer': f"{food_name.capitalize()} is a food item that can be part of a balanced diet.",
                    'confidence': 0.5,
                    'intent': intent
                }
                
        elif intent == "meal_planning":
            # Suggest food pairings
            # In a real implementation, you would use a food pairing database
            
            pairings = {
                "pizza": "Green salad, garlic bread, or a light soup.",
                "pasta": "Garlic bread, salad, or grilled vegetables.",
                "rice": "Stir-fried vegetables, curry, or grilled protein.",
                "bread": "Soup, salad, or as a sandwich with various fillings.",
                "apple": "Cheese, peanut butter, or as part of a fruit salad.",
                "banana": "Yogurt, oatmeal, or in a smoothie.",
                "salad": "Soup, sandwich, or as a side to a main protein dish.",
                "chicken": "Roasted vegetables, rice, or potatoes.",
                "beef": "Mashed potatoes, roasted vegetables, or a side salad.",
                "steak": "Baked potato, grilled asparagus, or sautéed mushrooms.",
                "fish": "Rice, steamed vegetables, or a light salad.",
                "salmon": "Asparagus, quinoa, or roasted vegetables.",
                "potato": "Grilled meat, vegetables, or as part of a vegetable medley."
            }
            
            if food_name.lower() in pairings:
                return {
                    'answer': f"{food_name.capitalize()} pairs well with: {pairings[food_name.lower()]}",
                    'confidence': 0.8,
                    'intent': intent
                }
            else:
                # Generic response
                return {
                    'answer': f"{food_name.capitalize()} can be paired with complementary flavors and textures. Consider balancing with vegetables, grains, or proteins as needed for a complete meal.",
                    'confidence': 0.5,
                    'intent': intent
                }
                
        elif intent == "cooking_method":
            # Provide cooking method information
            # In a real implementation, you would detect this from the image or use a more sophisticated approach
            
            cooking_methods = {
                "pizza": "baked in an oven",
                "pasta": "boiled in water",
                "rice": "simmered in water or broth",
                "bread": "baked in an oven",
                "apple": "raw (can also be baked, stewed, or sautéed)",
                "banana": "raw",
                "salad": "raw, with mixed ingredients",
                "chicken": "roasted, grilled, or fried",
                "beef": "grilled, roasted, or pan-seared",
                "steak": "grilled, pan-seared, or broiled",
                "fish": "baked, grilled, or pan-fried",
                "salmon": "baked, grilled, or pan-seared",
                "potato": "baked, boiled, or fried"
            }
            
            if food_name.lower() in cooking_methods:
                return {
                    'answer': f"{food_name.capitalize()} is typically {cooking_methods[food_name.lower()]}.",
                    'confidence': 0.7,
                    'intent': intent
                }
            else:
                # Generic response
                return {
                    'answer': f"I can't determine the exact cooking method for this {food_name} from the image alone.",
                    'confidence': 0.5,
                    'intent': intent
                }
                
        else:  # "other" or unknown intent
            return {
                'answer': f"I see {food_name} in the image. To learn more about it, you can ask about its nutrition, portion size, or how it fits into your diet.",
                'confidence': 0.6,
                'intent': "general_info"
            }


class FoodQuerySystem:
    """Main system for handling food and nutrition queries"""
    
    def __init__(self):
        from utils.nutrition_db import NutritionDatabase, NutritionAPI
        
        # Initialize components
        self.nutrition_db = NutritionDatabase()
        self.nutrition_api = NutritionAPI(local_db=self.nutrition_db)
        self.query_processor = QueryProcessor(self.nutrition_api)
        
        # Populate demo data
        self.nutrition_db.populate_demo_data()
    
    def process_query(self, query, image_path=None, detected_food=None, estimated_weight=None):
        """
        Process a food query with context from an image
        
        Args:
            query: Text query from user
            image_path: Path to food image (optional)
            detected_food: Pre-detected food class (optional)
            estimated_weight: Pre-estimated food weight in grams (optional)
            
        Returns:
            response: Dictionary with answer and relevant data
        """
        # Process the query using the QueryProcessor
        response = self.query_processor.process_query(
            query=query,
            detected_food=detected_food,
            weight_grams=estimated_weight
        )
        
        return response


if __name__ == "__main__":
    # Example usage
    query_system = FoodQuerySystem()
    
    # Test with different queries
    test_queries = [
        "How many calories are in this pizza?",
        "Is this apple healthy?",
        "How much protein is in this chicken?",
        "Does this pasta contain gluten?",
        "What goes well with salmon?",
        "How was this steak cooked?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = query_system.process_query(
            query=query,
            detected_food="pizza" if "pizza" in query else 
                         "apple" if "apple" in query else
                         "chicken" if "chicken" in query else
                         "pasta" if "pasta" in query else
                         "salmon" if "salmon" in query else
                         "steak" if "steak" in query else "unknown",
            estimated_weight=150
        )
        print(f"Answer: {response['answer']}")
        print(f"Intent: {response['intent']} (confidence: {response['confidence']})")
