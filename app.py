from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import uuid
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = None
if os.getenv('OPENAI_API_KEY'):
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# In-memory storage for chat sessions (in production, use a database)
chat_sessions = {}

# UDL Framework Knowledge Base
UDL_KNOWLEDGE = {
    "principles": {
        "principle_1": {
            "name": "Multiple Means of Engagement",
            "description": "Provide multiple ways for learners to engage with content and stay motivated",
            "guidelines": [
                "Provide options for recruiting interest",
                "Provide options for sustaining effort and persistence", 
                "Provide options for self-regulation"
            ]
        },
        "principle_2": {
            "name": "Multiple Means of Representation",
            "description": "Present information and content in multiple ways",
            "guidelines": [
                "Provide options for perception",
                "Provide options for language and symbols",
                "Provide options for comprehension"
            ]
        },
        "principle_3": {
            "name": "Multiple Means of Action and Expression",
            "description": "Provide multiple ways for learners to act and express what they know",
            "guidelines": [
                "Provide options for physical action",
                "Provide options for expression and communication",
                "Provide options for executive functions"
            ]
        }
    },
    "assessment_guidelines": [
        "Ensure assessments measure learning objectives, not access barriers",
        "Provide multiple ways for students to demonstrate knowledge",
        "Use clear, accessible language and instructions",
        "Offer flexible timing and pacing options",
        "Include culturally responsive content and examples",
        "Provide scaffolding and support structures",
        "Allow for student choice in how to express learning"
    ]
}

def generate_session_token():
    """Generate a unique session token"""
    return str(uuid.uuid4())

def get_or_create_session(session_token):
    """Get existing session or create new one"""
    if session_token not in chat_sessions:
        chat_sessions[session_token] = {
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }
    return chat_sessions[session_token]

def update_session_activity(session_token):
    """Update last activity timestamp"""
    if session_token in chat_sessions:
        chat_sessions[session_token]["last_activity"] = datetime.now().isoformat()

def analyze_assessment_udl(assessment_description):
    """Analyze assessment against UDL principles"""
    prompt = f"""
    You are a UDL (Universal Design for Learning) expert helping K-12 teachers create inclusive assessments. 
    
    Analyze the following assessment description against UDL principles and provide detailed feedback:
    
    Assessment: {assessment_description}
    
    Please provide your response in markdown format with the following structure:
    
    ## Assessment Analysis
    
    ### Strengths
    - List UDL-aligned elements that are already present
    
    ### Barriers Identified
    - List specific barriers to accessibility and inclusion
    
    ### Improvement Suggestions
    - Provide specific, actionable recommendations with UDL rationale
    - Include multiple options rather than prescriptive solutions
    
    ### Relevant UDL Principles
    - Explain which UDL principles (1, 2, or 3) are most relevant to address
    
    Focus on:
    - Removing barriers rather than accommodating differences
    - Building teacher understanding of UDL principles
    - Cultural responsiveness and student-specific factors
    - Cognitive load, accessibility, and equity considerations
    
    Be transparent about your UDL reasoning and present options rather than prescriptive solutions.
    """
    
    try:
        if not client:
            return "OpenAI API key not configured. Please add your API key to the .env file."
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing assessment: {str(e)}"

def generate_assessment_options(learning_objectives, grade_level, subject):
    """Generate multiple UDL-aligned assessment options"""
    prompt = f"""
    You are a UDL expert helping K-12 teachers design inclusive assessments. 
    
    Generate 4-5 diverse assessment options for:
    - Learning Objectives: {learning_objectives}
    - Grade Level: {grade_level}
    - Subject: {subject}
    
    Please provide your response in markdown format with the following structure:
    
    ## Assessment Design Options
    
    ### Option 1: [Format Name]
    **Format Description:** Clear description of the assessment format
    
    **Implementation Guidance:** Step-by-step instructions for teachers
    
    **UDL Rationale:** Why this option supports inclusive learning
    
    **Adaptation Considerations:** How teachers can customize for their context
    
    ### Option 2: [Format Name]
    [Continue with same structure for each option]
    
    Ensure all options:
    - Maintain equivalent rigor and learning objectives
    - Align with UDL Principle 3 (Multiple Means of Action and Expression)
    - Include diverse formats (written, oral, visual, multimedia, collaborative)
    - Support cultural responsiveness
    - Build teacher capacity for inclusive design
    
    Present options as choices rather than prescriptive solutions.
    """
    
    try:
        if not client:
            return "OpenAI API key not configured. Please add your API key to the .env file."
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating assessment options: {str(e)}"

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        session_token = data.get('session_token')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Get or create session
        if not session_token:
            session_token = generate_session_token()
        
        session = get_or_create_session(session_token)
        
        # Add user message to session
        session['messages'].append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Determine response type and generate appropriate response
        response_content = ""
        
        if any(keyword in message.lower() for keyword in ['evaluate', 'analyze', 'review', 'feedback']):
            # Assessment evaluation
            response_content = analyze_assessment_udl(message)
        elif any(keyword in message.lower() for keyword in ['create', 'design', 'generate', 'options', 'alternatives']):
            # Assessment design support
            # Extract learning objectives, grade level, and subject from message
            # For now, use the entire message as context
            response_content = generate_assessment_options(message, "K-12", "General")
        else:
            # General UDL guidance
            response_content = f"""# Welcome to the UDL Assessment Assistant!

I'm here to help you create more inclusive assessments using Universal Design for Learning principles.

## How I Can Help

### 1. Assessment Evaluation
Share an existing assessment and I'll analyze it against UDL principles, identifying:
- **Strengths**: UDL-aligned elements already present
- **Barriers**: Accessibility and inclusion issues
- **Improvement Suggestions**: Specific recommendations with UDL rationale
- **UDL Principles**: Which principles are most relevant to address

### 2. Assessment Design Support
Tell me about learning objectives, grade level, and subject, and I'll generate multiple UDL-aligned assessment options with:
- **Format Descriptions**: Clear assessment formats
- **Implementation Guidance**: Step-by-step instructions
- **UDL Rationale**: Why each option supports inclusive learning
- **Adaptation Considerations**: How to customize for your context

## Getting Started

You can:
- Describe an assessment you'd like me to evaluate
- Ask me to help design new assessment options
- Ask questions about UDL principles and inclusive assessment practices

Remember, I'm here to build your capacity for inclusive design while maintaining your professional agency as an educator."""
        
        # Add assistant response to session
        session['messages'].append({
            'role': 'assistant',
            'content': response_content,
            'timestamp': datetime.now().isoformat()
        })
        
        update_session_activity(session_token)
        
        return jsonify({
            'response': response_content,
            'session_token': session_token,
            'message_count': len(session['messages'])
        })
        
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/session/<session_token>', methods=['GET'])
def get_session(session_token):
    """Get chat session history"""
    if session_token not in chat_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = chat_sessions[session_token]
    return jsonify({
        'session_token': session_token,
        'messages': session['messages'],
        'created_at': session['created_at'],
        'last_activity': session['last_activity']
    })

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions (for debugging)"""
    return jsonify({
        'sessions': list(chat_sessions.keys()),
        'count': len(chat_sessions)
    })

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_sessions': len(chat_sessions)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)

