import random
import time
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.chat.util import Chat, reflections
import openai  # For GPT-3 integration (optional)

# Initialize Flask app
app = Flask(__name__)

# Download NLTK data
nltk.download('punkt')

class EmotionAIChatbot:
    def __init__(self):
        # Initialize emotion and sentiment analyzers
        self.emotion_classifier = pipeline(
            "text-classification", 
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            return_all_scores=True
        )
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize learning context
        self.learning_context = {
            'current_subject': None,
            'difficulty_level': 'beginner',
            'last_emotion': 'neutral',
            'consecutive_negative': 0
        }
        
        # Enhanced response patterns
        self.response_strategies = {
            'happy': {
                'responses': [
                    "I'm thrilled you're feeling good! What would you like to learn next?",
                    "Your positive energy is contagious! Want to tackle something new?"
                ],
                'actions': ['suggest_challenge', 'continue_learning']
            },
            'sad': {
                'responses': [
                    "I sense you might be feeling down. Would you like to take a break?",
                    "Learning can be tough sometimes. How about we try an easier topic?"
                ],
                'actions': ['offer_break', 'simplify_content', 'share_motivational_quote']
            },
            'angry': {
                'responses': [
                    "I hear your frustration. Let's take a deep breath together.",
                    "This seems frustrating. Would explaining it differently help?"
                ],
                'actions': ['calming_exercise', 'rephrase_explanation', 'switch_topic']
            },
            'confused': {
                'responses': [
                    "This seems confusing. Should I explain it differently?",
                    "Let me break this down into simpler parts."
                ],
                'actions': ['simplify_language', 'provide_example', 'use_analogy']
            },
            'neutral': {
                'responses': [
                    "How can I assist with your learning today?",
                    "What topic would you like to explore?"
                ],
                'actions': ['general_prompt']
            }
        }
        
        # Study tips database
        self.study_tips = {
            'math': ["Try breaking problems into smaller steps", "Practice with real-world examples"],
            'science': ["Create concept maps to visualize relationships", "Try teaching the concept to someone else"],
            'history': ["Create timelines to see the big picture", "Connect events to current affairs"]
        }
        
        # Motivational quotes
        self.motivational_quotes = [
            "Every expert was once a beginner. Keep going!",
            "Mistakes are proof you're trying. Don't give up!",
            "The more you learn, the more you earn - in knowledge!",
            "You don't have to be perfect, just persistent."
        ]
    
    def detect_emotion(self, text):
        """Enhanced emotion detection combining multiple approaches"""
        # Get emotion probabilities
        emotion_results = self.emotion_classifier(text)
        emotions = {item['label']: item['score'] for item in emotion_results[0]}
        
        # Get sentiment scores
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        
        # Determine primary emotion
        if sentiment['compound'] < -0.6:
            self.learning_context['consecutive_negative'] += 1
            if self.learning_context['consecutive_negative'] > 2:
                return 'frustrated'
            return 'angry'
        elif emotions['sadness'] > 0.8:
            self.learning_context['consecutive_negative'] += 1
            return 'sad'
        elif emotions['joy'] > 0.7:
            self.learning_context['consecutive_negative'] = 0
            return 'happy'
        elif emotions['surprise'] > 0.6:
            return 'confused'
        else:
            self.learning_context['consecutive_negative'] = max(0, self.learning_context['consecutive_negative'] - 1)
            return 'neutral'
    
    def generate_response(self, message, detected_emotion):
        """Generate context-aware, emotionally appropriate response"""
        # Update context
        self.learning_context['last_emotion'] = detected_emotion
        
        # Get base response
        strategy = self.response_strategies.get(detected_emotion, self.response_strategies['neutral'])
        base_response = random.choice(strategy['responses'])
        
        # Add emotional support elements
        support_elements = []
        for action in strategy['actions']:
            if action == 'share_motivational_quote':
                support_elements.append(f"\nRemember: {random.choice(self.motivational_quotes)}")
            elif action == 'offer_break' and detected_emotion in ['sad', 'angry']:
                support_elements.append("\nWould you like to try a 2-minute breathing exercise?")
            elif action == 'suggest_challenge' and detected_emotion == 'happy':
                support_elements.append("\nYou seem ready for a challenge! Want to try an advanced problem?")
            elif action == 'provide_example' and detected_emotion == 'confused':
                if self.learning_context['current_subject']:
                    support_elements.append(f"\nHere's an example about {self.learning_context['current_subject']}: ...")
        
        # Add study tips if relevant
        if self.learning_context['current_subject']:
            subject = self.learning_context['current_subject'].lower()
            if subject in self.study_tips:
                support_elements.append(f"\nStudy tip: {random.choice(self.study_tips[subject])}")
        
        # Combine all elements
        full_response = base_response + "".join(support_elements)
        return full_response
    
    def process_message(self, message):
        """Main processing pipeline"""
        # Detect emotion
        emotion = self.detect_emotion(message)
        
        # Generate response
        response = self.generate_response(message, emotion)
        
        # Return structured data
        return {
            'user_message': message,
            'bot_response': response,
            'detected_emotion': emotion,
            'context': self.learning_context
        }

# Initialize chatbot
chatbot = EmotionAIChatbot()

# Web Interface Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    response = chatbot.process_message(user_message)
    return jsonify(response)
if __name__ == '__main__':
    app.run(debug=True)
