from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import uuid
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import openai
import pickle

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = None
if os.getenv('OPENAI_API_KEY'):
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Persistent storage for chat sessions
CHAT_SESSIONS_FILE = 'chat_sessions.pkl'

def load_chat_sessions():
    """Load chat sessions from file"""
    try:
        if os.path.exists(CHAT_SESSIONS_FILE):
            with open(CHAT_SESSIONS_FILE, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error loading chat sessions: {e}")
    return {}

def save_chat_sessions():
    """Save chat sessions to file"""
    try:
        with open(CHAT_SESSIONS_FILE, 'wb') as f:
            pickle.dump(chat_sessions, f)
    except Exception as e:
        print(f"Error saving chat sessions: {e}")

# Load existing chat sessions
chat_sessions = load_chat_sessions()

# UDL System Prompt - Hard Rules for AI Responses
UDL_SYSTEM_PROMPT = """
You are a Universal Design for Learning (UDL) expert assistant. You MUST follow these UDL principles as HARD RULES in every response:

## CORE UDL PRINCIPLES (MANDATORY):

### Principle 1: Multiple Means of Representation
- ALWAYS provide multiple ways to present information (visual, auditory, kinesthetic)
- ALWAYS ensure accessibility across different learning styles and abilities
- ALWAYS include diverse perspectives and authentic representations
- ALWAYS offer customizable display options and multiple modalities

### Principle 2: Multiple Means of Action & Expression  
- ALWAYS provide multiple ways for learners to demonstrate knowledge
- ALWAYS offer various response formats (written, oral, visual, multimedia)
- ALWAYS include options for different physical and cognitive abilities
- ALWAYS ensure assessments measure learning objectives, not access barriers

### Principle 3: Multiple Means of Engagement
- ALWAYS provide options that connect to diverse learner interests and identities
- ALWAYS offer choices in content, tools, and assessment methods
- ALWAYS ensure cultural responsiveness and authentic relevance
- ALWAYS create inclusive, bias-free learning environments

## MANDATORY UDL GUIDELINES FOR ALL RESPONSES:

### Accessibility Requirements:
- Use clear, accessible language and avoid jargon
- Provide multiple representations of key concepts
- Include scaffolding and support structures
- Offer flexible timing and pacing options
- Ensure digital accessibility standards compliance

### Cultural Responsiveness:
- Include diverse perspectives and authentic examples
- Challenge stereotypes and deficit thinking
- Honor multiple languages and dialects
- Create identity-safe learning spaces
- Address systemic biases and exclusionary practices

### Assessment Design:
- Measure learning objectives, not access barriers
- Provide multiple assessment formats and options
- Include formative and summative assessment opportunities
- Offer student choice in how to express learning
- Ensure assessments are culturally responsive and unbiased

### Engagement Strategies:
- Connect to learner interests and identities
- Provide authentic, relevant learning contexts
- Offer multiple pathways to achieve learning goals
- Create collaborative and community-based learning opportunities
- Nurture joy, play, and intrinsic motivation

## RESPONSE REQUIREMENTS:
1. Every response MUST reference relevant UDL principles
2. Every response MUST provide multiple options, not single solutions
3. Every response MUST address accessibility and inclusion
4. Every response MUST be culturally responsive and bias-aware
5. Every response MUST support teacher agency while building UDL capacity

## FORBIDDEN PRACTICES:
- Never provide single, prescriptive solutions
- Never ignore accessibility considerations
- Never use deficit-based language or assumptions
- Never create barriers for learners with disabilities
- Never perpetuate cultural biases or stereotypes

Remember: UDL is about designing for ALL learners from the start, not retrofitting accommodations. Every recommendation must be inclusive, flexible, and culturally responsive.
"""

# UDL Framework Knowledge Base (Enhanced)
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
            "conversation_state": "waiting_for_objectives",
            "context": {
                "learning_objectives": None,
                "grade_level": None,
                "subject": None,
                "class_size": None,
                "time_constraints": None,
                "student_needs": None,
                "cultural_context": None,
                "technology_access": None,
                "assessment_preferences": None
            },
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }
    return chat_sessions[session_token]

def update_session_activity(session_token):
    """Update last activity timestamp"""
    if session_token in chat_sessions:
        chat_sessions[session_token]["last_activity"] = datetime.now().isoformat()
        save_chat_sessions()

def analyze_assessment_udl(assessment_description):
    """Analyze assessment against UDL principles"""
    user_prompt = f"""
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
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": UDL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1200,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing assessment: {str(e)}"

def generate_assessment_options(learning_objectives, grade_level, subject):
    """Generate multiple UDL-aligned assessment options"""
    user_prompt = f"""
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
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": UDL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1200,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating assessment options: {str(e)}"

def process_objectives_input(session, user_message):
    """Process learning objectives input and extract key information"""
    user_prompt = f"""
    A teacher has provided their learning objectives: "{user_message}"
    
    Extract and structure the following information:
    1. Learning objectives (list them clearly)
    2. Subject area (if mentioned)
    3. Grade level (if mentioned)
    4. Any other context clues
    
    Respond in this format:
    OBJECTIVES: [list the objectives]
    SUBJECT: [subject if mentioned, otherwise "Not specified"]
    GRADE: [grade level if mentioned, otherwise "Not specified"]
    ADDITIONAL_CONTEXT: [any other relevant information]
    """
    
    try:
        if not client:
            return "OpenAI API key not configured. Please add your API key to the .env file."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": UDL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing objectives: {str(e)}"

def generate_clarifying_questions(session):
    """Generate contextual clarifying questions based on gathered information"""
    context = session["context"]
    
    # Build context summary
    context_summary = f"""
    Current context:
    - Learning Objectives: {context['learning_objectives'] or 'Not provided'}
    - Subject: {context['subject'] or 'Not specified'}
    - Grade Level: {context['grade_level'] or 'Not specified'}
    - Class Size: {context['class_size'] or 'Not specified'}
    - Time Constraints: {context['time_constraints'] or 'Not specified'}
    - Student Needs: {context['student_needs'] or 'Not specified'}
    - Cultural Context: {context['cultural_context'] or 'Not specified'}
    - Technology Access: {context['technology_access'] or 'Not specified'}
    - Assessment Preferences: {context['assessment_preferences'] or 'Not specified'}
    """
    
    user_prompt = f"""
    Based on the current context, generate ONE short, focused clarifying question to help create better UDL assessments.
    Focus on the most important missing information for designing inclusive assessments.
    
    {context_summary}
    
    Generate a question that:
    1. Addresses the biggest gap in the context above
    2. Is short and easy to answer
    3. Focuses on student diversity and needs
    4. Is practical for assessment design
    
    Return ONLY the question, no extra text or formatting.
    """
    
    try:
        if not client:
            return "OpenAI API key not configured. Please add your API key to the .env file."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": UDL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating clarifying questions: {str(e)}"

def generate_udl_assessments_with_rubric(session):
    """Generate UDL assessments and common rubric based on gathered context"""
    context = session["context"]
    
    context_summary = f"""
    Complete context for assessment design:
    - Learning Objectives: {context['learning_objectives']}
    - Subject: {context['subject']}
    - Grade Level: {context['grade_level']}
    - Class Size: {context['class_size']}
    - Time Constraints: {context['time_constraints']}
    - Student Needs: {context['student_needs']}
    - Cultural Context: {context['cultural_context']}
    - Technology Access: {context['technology_access']}
    - Assessment Preferences: {context['assessment_preferences']}
    """
    
    user_prompt = f"""
    Generate 4-5 diverse UDL-aligned assessment options and one comprehensive rubric that works for all assessments.
    
    {context_summary}
    
    Structure your response as follows:
    
    ## UDL Assessment Options
    
    ### Assessment Option 1: [Name]
    **Format:** [Description]
    **UDL Alignment:** [Which principles this addresses]
    **Implementation:** [Step-by-step guidance]
    **Accessibility Features:** [How this removes barriers]
    
    ### Assessment Option 2: [Name]
    [Continue for each option]
    
    ## Common Assessment Rubric
    
    ### [Criteria 1 Name]
    **Excellent (4):** [Description]
    **Proficient (3):** [Description] 
    **Developing (2):** [Description]
    **Beginning (1):** [Description]
    
    ### [Criteria 2 Name]
    [Continue for each criteria]
    
    ## Implementation Notes
    - How to use the rubric across all assessments
    - Adaptation suggestions for different learners
    - Cultural responsiveness considerations
    
    Ensure all assessments:
    - Measure the same learning objectives
    - Provide multiple means of expression
    - Are culturally responsive and accessible
    - Remove barriers rather than create accommodations
    """
    
    try:
        if not client:
            return "OpenAI API key not configured. Please add your API key to the .env file."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": UDL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating assessments: {str(e)}"

def get_general_udl_guidance(user_message):
    """Provide general UDL guidance using system prompt"""
    user_prompt = f"""
    The user has asked: "{user_message}"
    
    Provide comprehensive UDL guidance that addresses their question while following all UDL principles.
    
    Structure your response to include:
    - Direct answer to their question
    - Multiple options and approaches (never single solutions)
    - Accessibility considerations
    - Cultural responsiveness factors
    - Relevant UDL principles and guidelines
    - Practical implementation strategies
    - Resources for further learning
    
    Ensure your response builds teacher capacity while maintaining their professional agency.
    """
    
    try:
        if not client:
            return "OpenAI API key not configured. Please add your API key to the .env file."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": UDL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error providing UDL guidance: {str(e)}"

def extract_context_from_response(response_text, context_field):
    """Extract specific context information from AI response"""
    lines = response_text.split('\n')
    for line in lines:
        if line.startswith(context_field.upper() + ':'):
            return line.split(':', 1)[1].strip()
    return None

def update_context_from_response(session, response_text):
    """Update session context based on AI response"""
    context = session["context"]
    
    # Extract information from the response
    objectives = extract_context_from_response(response_text, "OBJECTIVES")
    subject = extract_context_from_response(response_text, "SUBJECT")
    grade = extract_context_from_response(response_text, "GRADE")
    
    if objectives:
        context["learning_objectives"] = objectives
    if subject and subject != "Not specified":
        context["subject"] = subject
    if grade and grade != "Not specified":
        context["grade_level"] = grade

def handle_context_gathering(session, user_message):
    """Handle context gathering phase - extract information from user responses"""
    # This is a simplified approach - in a real implementation, you might want
    # more sophisticated NLP to extract specific information
    context = session["context"]
    
    # Simple keyword-based context extraction
    user_lower = user_message.lower()
    
    # Extract grade level
    if any(grade in user_lower for grade in ['kindergarten', 'k-', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th', '11th', '12th']):
        context["grade_level"] = user_message
    
    # Extract class size
    if 'students' in user_lower or 'class' in user_lower:
        context["class_size"] = user_message
    
    # Extract time constraints
    if any(time_word in user_lower for time_word in ['minutes', 'hours', 'days', 'weeks', 'time']):
        context["time_constraints"] = user_message
    
    # Extract student needs
    if any(need in user_lower for need in ['special', 'disability', 'iep', '504', 'ell', 'esl', 'gifted']):
        context["student_needs"] = user_message
    
    # Extract cultural context
    if any(cultural in user_lower for cultural in ['culture', 'diverse', 'multicultural', 'language', 'background']):
        context["cultural_context"] = user_message
    
    # Extract technology access
    if any(tech in user_lower for tech in ['computer', 'tablet', 'internet', 'technology', 'digital']):
        context["technology_access"] = user_message
    
    # Extract assessment preferences
    if any(assessment in user_lower for assessment in ['test', 'project', 'presentation', 'portfolio', 'essay']):
        context["assessment_preferences"] = user_message
    
    # Check if we have enough context to proceed
    required_fields = ["learning_objectives", "grade_level", "subject"]
    filled_fields = sum(1 for field in required_fields if context[field])
    total_fields = len(required_fields)
    
    if filled_fields >= total_fields:
        return "ready_for_assessments"
    else:
        return "need_more_context"

def get_context_progress(session):
    """Get progress indicator for context gathering"""
    context = session["context"]
    required_fields = ["learning_objectives", "grade_level", "subject"]
    filled_fields = sum(1 for field in required_fields if context[field])
    total_fields = len(required_fields)
    return filled_fields, total_fields

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint with structured conversation flow"""
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
        conversation_state = session['conversation_state']
        
        # Add user message to session
        session['messages'].append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        save_chat_sessions()
        
        # Handle conversation flow based on current state
        response_content = ""
        
        if conversation_state == "waiting_for_objectives":
            # Process learning objectives input
            objectives_response = process_objectives_input(session, message)
            update_context_from_response(session, objectives_response)
            
            # Generate first clarifying question
            question = generate_clarifying_questions(session)
            
            # Update state
            session['conversation_state'] = "gathering_context"
            
            filled, total = get_context_progress(session)
            response_content = f"""# Learning Objectives Received ✅

{objectives_response}

---

## Quick Question ({filled}/{total})

To design the best assessments for your students, I need to ask:

**{question}**

Just type your answer! 😊"""
        
        elif conversation_state == "gathering_context":
            # Handle context gathering
            context_status = handle_context_gathering(session, message)
            
            if context_status == "ready_for_assessments":
                # Generate assessments and rubric
                assessments_response = generate_udl_assessments_with_rubric(session)
                session['conversation_state'] = "completed"
                
                response_content = f"""# UDL Assessment Options & Rubric 🎯

Based on all the information you've provided, here are your personalized UDL assessment options:

{assessments_response}

## Session Complete ✅

You now have multiple UDL-aligned assessment options with a common rubric. You can start a new session anytime by sharing new learning objectives!"""
            
            else:
                # Need more context - generate one more question
                question = generate_clarifying_questions(session)
                filled, total = get_context_progress(session)
                response_content = f"""Thanks! 👍

## Quick Question ({filled}/{total})

**{question}**

Almost done! 🚀"""
        
        elif conversation_state == "completed":
            # Session completed, offer to start new session
            response_content = """# Session Complete ✅

This session is complete! You have your UDL assessment options and rubric.

To start a new assessment design session, simply share your new learning objectives and I'll guide you through the process again.

**Example:** "My learning objectives are for students to understand photosynthesis, identify plant parts, and explain the process to others."

Ready for your next lesson's assessments? 🚀"""
        
        else:
            # Fallback to general guidance
            response_content = get_general_udl_guidance(message)
        
        # Add assistant response to session
        session['messages'].append({
            'role': 'assistant',
            'content': response_content,
            'timestamp': datetime.now().isoformat()
        })
        
        update_session_activity(session_token)
        save_chat_sessions()
        
        return jsonify({
            'response': response_content,
            'session_token': session_token,
            'conversation_state': session['conversation_state'],
            'message_count': len(session['messages']),
            'context_summary': {
                'has_objectives': bool(session['context']['learning_objectives']),
                'has_grade_level': bool(session['context']['grade_level']),
                'has_subject': bool(session['context']['subject'])
            }
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

@app.route('/api/session/<session_token>/reset', methods=['POST'])
def reset_session(session_token):
    """Reset a session to start over"""
    if session_token not in chat_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    # Reset session to initial state
    chat_sessions[session_token] = {
        "messages": [],
        "conversation_state": "waiting_for_objectives",
        "context": {
            "learning_objectives": None,
            "grade_level": None,
            "subject": None,
            "class_size": None,
            "time_constraints": None,
            "student_needs": None,
            "cultural_context": None,
            "technology_access": None,
            "assessment_preferences": None
        },
        "created_at": datetime.now().isoformat(),
        "last_activity": datetime.now().isoformat()
    }
    
    save_chat_sessions()
    return jsonify({
        'message': 'Session reset successfully',
        'session_token': session_token,
        'conversation_state': 'waiting_for_objectives'
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

