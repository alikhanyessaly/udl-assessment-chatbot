from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import uuid
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import openai
import pickle
import PyPDF2
import io

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
            "state": "Start",
            "branch": None,  # "design" or "evaluate"
            "context": {
                "learning_objectives": None,
                "grade_level": None,
                "subject": None,
                "assessment_content": None,
                "assessment_alignment": None  # "aligned" or "not_aligned"
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

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

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

# New State Machine Functions for UDL Assessment Assistant

def extract_basic_info(user_message):
    """Extract learning objectives, grade, and subject from user input"""
    user_prompt = f"""
    Extract the following information from this teacher input: "{user_message}"
    
    Format your response as:
    OBJECTIVES: [list learning objectives]
    GRADE: [grade level if mentioned, otherwise "Not specified"]
    SUBJECT: [subject if mentioned, otherwise "Not specified"]
    """
    
    try:
        if not client:
            return "OpenAI API key not configured. Please add your API key to the .env file."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts structured information from teacher input."},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error extracting information: {str(e)}"

def classify_assessment_udl_alignment(assessment_content):
    """Classify if assessment is UDL aligned or not"""
    user_prompt = f"""
    Analyze this assessment for UDL alignment:
    
    {assessment_content}
    
    Determine if this assessment is UDL-aligned or not. Consider:
    - Multiple means of representation
    - Multiple means of action and expression  
    - Multiple means of engagement
    - Accessibility barriers
    - Cultural responsiveness
    
    Respond with only: "ALIGNED" or "NOT_ALIGNED"
    """
    
    try:
        if not client:
            return "NOT_ALIGNED"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a UDL expert classifier. Respond with only ALIGNED or NOT_ALIGNED."},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=10,
            temperature=0.1
        )
        result = response.choices[0].message.content.strip().upper()
        return "aligned" if "ALIGNED" in result and "NOT" not in result else "not_aligned"
    except Exception as e:
        return "not_aligned"

def generate_udl_assessment_options(context):
    """Generate UDL assessment options and universal rubric"""
    user_prompt = f"""
    Generate UDL assessment options based on:
    - Learning Objectives: {context['learning_objectives']}
    - Grade: {context['grade_level']}
    - Subject: {context['subject']}
    
    Provide:
    1. Multiple assessment options with different formats
    2. A universal rubric that works for all options
    3. Implementation guidance
    
    Structure as:
    ## Assessment Options
    ### Option 1: [Name]
    **Format:** [Description]
    **UDL Alignment:** [Explanation]
    
    ## Universal Rubric
    [Rubric details]
    
    ## Implementation Notes
    [Guidance for teachers]
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
        return f"Error generating assessments: {str(e)}"

def evaluate_assessment_udl(assessment_content):
    """Evaluate assessment against UDL principles"""
    user_prompt = f"""
    Evaluate this assessment against UDL principles:

{assessment_content}

    Provide:
    1. Strengths (UDL-aligned elements)
    2. Barriers identified
    3. Improvement suggestions with UDL rationale
    4. Relevant UDL principles
    
    Focus on removing barriers rather than accommodating differences.
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
        return f"Error evaluating assessment: {str(e)}"

def generate_quality_check_report(assessments_markdown, context):
    """Generate a Quality Check report/checklist for produced assessments"""
    user_prompt = f"""
    You are performing a Quality Check on the following UDL assessment materials.

    ## CONTEXT
    - Learning Objectives: {context.get('learning_objectives')}
    - Grade Level: {context.get('grade_level')}
    - Subject: {context.get('subject')}

    ## ASSESSMENT MATERIALS
    {assessments_markdown}

    Produce a Quality Check report that includes:
    1) Validation: correctness, relevance to objectives, rigor appropriate to grade/subject
    2) Accessibility & UDL: representation, action/expression, engagement (note concrete strengths and gaps)
    3) Cultural Responsiveness: identity representation, bias checks, authenticity
    4) Practicality: resources/tools required, timing/pacing flexibility, assistive tech compatibility
    5) Checklist with [ ] items teachers can verify quickly
    6) Top 5 fixes to apply before classroom use

    Use clear headings and concise, actionable bullets. Keep structure consistent and teacher-ready.
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
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating quality check: {str(e)}"

def generate_iteration_refinements(assessments_markdown, qc_report, context):
    """Generate refinement suggestions and a concise finalize-ready summary"""
    user_prompt = f"""
    You are improving the following UDL assessment set based on the Quality Check.

    ## CONTEXT
    - Learning Objectives: {context.get('learning_objectives')}
    - Grade Level: {context.get('grade_level')}
    - Subject: {context.get('subject')}

    ## CURRENT ASSESSMENTS
    {assessments_markdown}

    ## QUALITY CHECK REPORT
    {qc_report}

    Provide:
    1) Iteration Plan: prioritized refinements mapped to QC issues (bulleted, specific)
    2) Updated Accessibility Notes: concrete adjustments (captioning, alternative formats, etc.)
    3) Cultural Responsiveness Enhancements: examples, resources, language guidance
    4) Finalize-ready Summary: short teacher-facing summary of the improved set

    Be concise, structured, and immediately actionable.
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
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating iteration refinements: {str(e)}"

def apply_user_refinements(assessments_markdown, user_suggestions, context):
    """Apply user-provided refinement suggestions to regenerate improved assessments"""
    user_prompt = f"""
    You are updating an existing UDL assessment set based on a teacher's refinement suggestions.

    ## CONTEXT
    - Learning Objectives: {context.get('learning_objectives')}
    - Grade Level: {context.get('grade_level')}
    - Subject: {context.get('subject')}

    ## CURRENT ASSESSMENTS
    {assessments_markdown}

    ## TEACHER SUGGESTIONS (APPLY THESE CHANGES NOW)
    {user_suggestions}

    Regenerate the full assessment set (4–5 options) and the universal rubric with the changes applied.
    Keep the structure consistent and ensure strong alignment to UDL (representation, action & expression, engagement), accessibility, and cultural responsiveness. Be concise and teacher-ready.
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
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error applying refinements: {str(e)}"
def parse_extracted_info(response_text):
    """Parse extracted information from AI response"""
    context = {"learning_objectives": None, "grade_level": None, "subject": None}
    
    lines = response_text.split('\n')
    for line in lines:
        if line.startswith('OBJECTIVES:'):
            context["learning_objectives"] = line.split(':', 1)[1].strip()
        elif line.startswith('GRADE:'):
            grade = line.split(':', 1)[1].strip()
            if grade != "Not specified":
                context["grade_level"] = grade
        elif line.startswith('SUBJECT:'):
            subject = line.split(':', 1)[1].strip()
            if subject != "Not specified":
                context["subject"] = subject
    
    return context

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle PDF file upload and extract text"""
    try:
        session_token = request.form.get('session_token')
        if not session_token:
            return jsonify({'error': 'Session token required'}), 400
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are supported'}), 400
        
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(file)
        
        # Store in session context
        session = get_or_create_session(session_token)
        session['context']['assessment_content'] = pdf_text
        save_chat_sessions()
        
        return jsonify({
            'success': True,
            'text': pdf_text[:500] + '...' if len(pdf_text) > 500 else pdf_text,
            'message': 'PDF uploaded and processed successfully'
        })
    
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint with new state machine logic"""
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
        current_state = session['state']
        
        # Add user message to session
        session['messages'].append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        save_chat_sessions()
        
        # Handle conversation flow based on current state
        response_content = ""
        
        # State: Start
        if current_state == "Start":
            if message.lower().strip() == "design":
                session['branch'] = 'design'
                session['state'] = 'DesignMode'
                response_content = """# Design Mode Selected 🎨

Could you provide:
1) Learning Objectives
2) Grade
3) Subject

Please share these details and I'll help you create UDL-aligned assessments!

---

Quick examples:

<button onclick=\"sendQuickMessage('Learning objectives: explain photosynthesis process; identify plant parts; present findings. Subject: Biology. Grade: 8')\" style=\"background: #4f46e5; color: white; border: none; padding: 10px 14px; border-radius: 8px; cursor: pointer; font-size: 0.95rem; margin-right: 8px;\">🌿 Biology - Photosynthesis (Grade 8)</button>

<button onclick=\"sendQuickMessage('Learning objectives: solve linear equations; graph solutions; justify reasoning. Subject: Mathematics. Grade: 9')\" style=\"background: #10b981; color: white; border: none; padding: 10px 14px; border-radius: 8px; cursor: pointer; font-size: 0.95rem;\">➗ Math - Linear Equations (Grade 9)</button>"""
            elif message.lower().strip() == "evaluate":
                session['branch'] = 'evaluate'
                session['state'] = 'EvaluateMode'
                response_content = """# Evaluation Mode Selected 🔍

Could you provide:
1) Learning Objectives
2) Grade
3) Subject

Please share these details and then upload your assessment for evaluation!

---

Quick examples:

<button onclick=\"sendQuickMessage('Learning objectives: analyze causes of World War II; evaluate primary sources; synthesize historical arguments. Subject: History. Grade: 10')\" style=\"background: #f59e0b; color: white; border: none; padding: 10px 14px; border-radius: 8px; cursor: pointer; font-size: 0.95rem; margin-right: 8px;\">📜 History - WWII Analysis (Grade 10)</button>

<button onclick=\"sendQuickMessage('Learning objectives: write persuasive essays; analyze rhetorical devices; present oral arguments. Subject: English Language Arts. Grade: 11')\" style=\"background: #8b5cf6; color: white; border: none; padding: 10px 14px; border-radius: 8px; cursor: pointer; font-size: 0.95rem;\">📚 ELA - Persuasive Writing (Grade 11)</button>"""
            else:
                response_content = """# Welcome to the UDL Assessment Assistant! 🎯

Please choose one of the following options:
- **Design** → Create new UDL-aligned assessments
- **Evaluate** → Analyze existing assessments for UDL compliance

Type "design" or "evaluate" to continue!"""
        
        # Design Branch States
        elif current_state == "DesignMode":
            # Extract basic info and move to next state
            info_response = extract_basic_info(message)
            extracted_context = parse_extracted_info(info_response)
            session['context'].update(extracted_context)
            session['state'] = 'DesignInputReceived'
            
            response_content = f"""# Information Received ✅

{info_response}

---

Do you have an assessment that you would like to create UDL options for?

<div style=\"margin-top: 8px; display: flex; gap: 8px;\">
  <button onclick=\"sendQuickMessage('Yes')\" style=\"background:#4f46e5;color:#fff;border:none;padding:8px 14px;border-radius:8px;cursor:pointer;\">Yes</button>
  <button onclick=\"sendQuickMessage('No')\" style=\"background:#ef4444;color:#fff;border:none;padding:8px 14px;border-radius:8px;cursor:pointer;\">No</button>
</div>"""
        
        elif current_state == "DesignInputReceived":
            if message.lower().strip() in ['yes', 'y', 'yeah', 'yep', 'sure']:
                session['state'] = 'DesignAssessmentYes'
                response_content = """# Assessment Upload Needed 📄

Please upload or describe the assessment you'd like to create UDL options for.

You can:
- Upload a file (PDF, Word, etc.)
- Describe the assessment in text
- Share the assessment questions and instructions"""
            else:
                # Generate assessment options and prompt user for refinement suggestions
                session['state'] = 'QualityCheckPhase'
                assessments = generate_udl_assessment_options(session['context'])
                session['context']['last_assessments'] = assessments
                session['context']['last_quality_check'] = None
                response_content = f"""# UDL Assessment Options & Rubric 🎯

{assessments}

---

Share any suggestions (e.g., add captions, broaden examples, adjust timing). I'll update the assessments. Type "finalize" when done."""
        
        elif current_state == "DesignAssessmentYes":
            # Check if assessment was uploaded via PDF or text
            if session['context'].get('assessment_content'):
                # Already uploaded via PDF endpoint
                session['state'] = 'DesignAssessmentReceived'
                response_content = """# Assessment Received ✅

Would you like to:

<div style=\"margin-top: 8px; display: flex; gap: 8px;\">
  <button onclick=\"sendQuickMessage('1')\" style=\"background:#4f46e5;color:#fff;border:none;padding:8px 14px;border-radius:8px;cursor:pointer;\">1) Evaluate this assessment</button>
  <button onclick=\"sendQuickMessage('2')\" style=\"background:#10b981;color:#fff;border:none;padding:8px 14px;border-radius:8px;cursor:pointer;\">2) Create alternative versions</button>
</div>"""
            else:
                # User provided text description
                session['context']['assessment_content'] = message
                session['state'] = 'DesignAssessmentReceived'
                response_content = """# Assessment Received ✅

Would you like to:

<div style=\"margin-top: 8px; display: flex; gap: 8px;\">
  <button onclick=\"sendQuickMessage('1')\" style=\"background:#4f46e5;color:#fff;border:none;padding:8px 14px;border-radius:8px;cursor:pointer;\">1) Evaluate this assessment</button>
  <button onclick=\"sendQuickMessage('2')\" style=\"background:#10b981;color:#fff;border:none;padding:8px 14px;border-radius:8px;cursor:pointer;\">2) Create alternative versions</button>
</div>"""
        
        elif current_state == "DesignAssessmentReceived":
            if message.strip() == "1":
                session['state'] = 'EvaluateAssessmentReceived'
                alignment = classify_assessment_udl_alignment(session['context']['assessment_content'])
                session['context']['assessment_alignment'] = alignment
                
                if alignment == "aligned":
                    session['state'] = 'EvaluateAligned'
                    # Generate detailed evaluation report even for aligned assessments
                    evaluation = evaluate_assessment_udl(session['context']['assessment_content'])
                    session['context']['evaluation_report'] = evaluation
                    response_content = f"""# UDL Assessment Evaluation Report 📊

{evaluation}

---

## ✅ This assignment is aligned with UDL guidelines!

Your assessment demonstrates strong UDL principles. Great job designing for all learners!"""
                else:
                    session['state'] = 'EvaluateNotAligned'
                    # Generate detailed evaluation report
                    evaluation = evaluate_assessment_udl(session['context']['assessment_content'])
                    session['context']['evaluation_report'] = evaluation
                    response_content = f"""# UDL Assessment Evaluation Report 📊

{evaluation}

---

## This assignment is not fully aligned with UDL guidelines.

Would you like to adapt this assessment to UDL?

<div style=\"margin-top: 8px; display: flex; gap: 8px;\">
  <button onclick=\"sendQuickMessage('Yes')\" style=\"background:#4f46e5;color:#fff;border:none;padding:8px 14px;border-radius:8px;cursor:pointer;\">Yes - Create UDL-aligned alternatives</button>
  <button onclick=\"sendQuickMessage('No')\" style=\"background:#ef4444;color:#fff;border:none;padding:8px 14px;border-radius:8px;cursor:pointer;\">No - Just review analysis</button>
</div>"""
            else:
                # Generate assessment options immediately
                session['state'] = 'QualityCheckPhase'
                assessments = generate_udl_assessment_options(session['context'])
                session['context']['last_assessments'] = assessments
                session['context']['last_quality_check'] = None
                response_content = f"""# UDL Assessment Options & Rubric 🎯

{assessments}

---

Share any suggestions (e.g., add captions, broaden examples, adjust timing). I'll update the assessments. Type "finalize" when done."""
        
        elif current_state == "DesignAssessmentNo":
            session['state'] = 'QualityCheckPhase'
            assessments = generate_udl_assessment_options(session['context'])
            response_content = f"""# UDL Assessment Options & Rubric 🎯

{assessments}

---

## Quality Check Phase ✅

Now performing Quality Check of materials and application in the classroom...

Please review the assessment options and rubric above."""
        
        # Evaluate Branch States
        elif current_state == "EvaluateMode":
            info_response = extract_basic_info(message)
            extracted_context = parse_extracted_info(info_response)
            session['context'].update(extracted_context)
            session['state'] = 'EvaluateInputReceived'
            
            response_content = f"""# Information Received ✅

{info_response}

---

Could you upload the assessment for evaluation?

You can:
- **Upload a PDF** (click the 📎 paperclip button)
- **Paste the text** directly in the chat

Please share your assessment content (questions, instructions, rubric, etc.)."""
        
        elif current_state == "EvaluateInputReceived":
            session['context']['assessment_content'] = message
            session['state'] = 'EvaluateAssessmentReceived'
            alignment = classify_assessment_udl_alignment(message)
            session['context']['assessment_alignment'] = alignment
            
            if alignment == "aligned":
                session['state'] = 'EvaluateAligned'
                # Generate detailed evaluation report even for aligned assessments
                evaluation = evaluate_assessment_udl(session['context']['assessment_content'])
                session['context']['evaluation_report'] = evaluation
                response_content = f"""# UDL Assessment Evaluation Report 📊

{evaluation}

---

## ✅ This assignment is aligned with UDL guidelines!

Your assessment demonstrates strong UDL principles. Great job designing for all learners!"""
            else:
                session['state'] = 'EvaluateNotAligned'
                # Generate detailed evaluation report
                evaluation = evaluate_assessment_udl(session['context']['assessment_content'])
                session['context']['evaluation_report'] = evaluation
                response_content = f"""# UDL Assessment Evaluation Report 📊

{evaluation}

---

## This assignment is not fully aligned with UDL guidelines.

Would you like to adapt this assessment to UDL?

<div style=\"margin-top: 8px; display: flex; gap: 8px;\">
  <button onclick=\"sendQuickMessage('Yes')\" style=\"background:#4f46e5;color:#fff;border:none;padding:8px 14px;border-radius:8px;cursor:pointer;\">Yes - Create UDL-aligned alternatives</button>
  <button onclick=\"sendQuickMessage('No')\" style=\"background:#ef4444;color:#fff;border:none;padding:8px 14px;border-radius:8px;cursor:pointer;\">No - Just review analysis</button>
</div>"""
        
        elif current_state == "EvaluateAligned":
            session['state'] = 'End'
            evaluation = evaluate_assessment_udl(session['context']['assessment_content'])
            response_content = f"""# UDL Assessment Evaluation Report 📊

{evaluation}

---

## Evaluation Complete ✅

Your assessment demonstrates strong UDL alignment. Great job designing for all learners!"""
        
        elif current_state == "EvaluateNotAligned":
            if message.lower().strip() in ['yes', 'y', 'yeah', 'yep', 'sure']:
                # Generate UDL-aligned assessment options immediately
                session['state'] = 'QualityCheckPhase'
                assessments = generate_udl_assessment_options(session['context'])
                session['context']['last_assessments'] = assessments
                session['context']['last_quality_check'] = None
                response_content = f"""# UDL Assessment Options & Rubric 🎯

{assessments}

---

Share any suggestions (e.g., add captions, broaden examples, adjust timing). I'll update the assessments. Type "finalize" when done."""
            else:
                session['state'] = 'ProvideNotAlignedReason'
                evaluation = evaluate_assessment_udl(session['context']['assessment_content'])
                response_content = f"""# Assessment Analysis Report 📊

{evaluation}

---

## Analysis Complete ✅

Here's why this assignment is not aligned with UDL guidelines: [analysis]"""
        
        elif current_state == "ProvideNotAlignedReason":
            session['state'] = 'End'
            response_content = """# Session Complete ✅

Your assessment analysis is complete. You now have detailed feedback on UDL alignment and improvement suggestions.

To start a new session, type "design" or "evaluate"."""
        
        # Shared States
        elif current_state == "AssessmentCreationMode":
            session['state'] = 'QualityCheckPhase'
            assessments = generate_udl_assessment_options(session['context'])
            # store latest assessments
            session['context']['last_assessments'] = assessments
            session['context']['last_quality_check'] = None
            response_content = f"""# UDL Assessment Options & Rubric 🎯

{assessments}

---

Share any suggestions (e.g., add captions, broaden examples, adjust timing). I'll update the assessments. Type "finalize" when done."""
        
        elif current_state == "QualityCheckPhase":
            # Interactive refinement loop: accept suggestions and regenerate
            lower = message.lower().strip()
            if lower in ['finalize', 'done', 'finish', 'no more changes']:
                session['state'] = 'End'
                response_content = """# Session Finalized ✅

Your UDL assessment set is ready. To start a new session, type "design" or "evaluate"."""
            elif lower in ['', 'continue', 'next', 'ok']:
                # Prompt for suggestions (no automatic QC dump)
                response_content = """# Quality Check - Your Suggestions ✍️

Share any suggestions (e.g., add captions, broaden examples, adjust timing). I'll update the assessments. Type "finalize" when done."""
            else:
                # Treat message as refinement suggestions and regenerate
                assessments_md = session['context'].get('last_assessments') or "(no assessments cached)"
                refined = apply_user_refinements(assessments_md, message, session['context'])
                session['context']['last_assessments'] = refined
                response_content = f"""# Updated UDL Assessment Options & Rubric 🔄

{refined}

---

Provide more suggestions to iterate, or type "finalize" to finish."""
        
        elif current_state == "IterationPhase":
            # Mirror QualityCheckPhase behavior to allow continued iteration
            lower = message.lower().strip()
            if lower in ['finalize', 'done', 'finish', 'no more changes']:
                session['state'] = 'End'
                response_content = """# Session Finalized ✅

Your UDL assessment set is ready. To start a new session, type "design" or "evaluate"."""
            else:
                assessments_md = session['context'].get('last_assessments') or "(no assessments cached)"
                refined = apply_user_refinements(assessments_md, message, session['context'])
                session['context']['last_assessments'] = refined
                response_content = f"""# Updated UDL Assessment Options & Rubric 🔄

{refined}

---

Provide more suggestions to iterate, or type "finalize" to finish."""
        
        # End State
        elif current_state == "End":
            if message.lower().strip() in ['design', 'evaluate']:
                # Reset session and start new branch
                session['state'] = 'Start'
                session['branch'] = None
                session['context'] = {
                    "learning_objectives": None,
                    "grade_level": None,
                    "subject": None,
                    "assessment_content": None,
                    "assessment_alignment": None
                }
                # Recursive call to handle the new mode selection
                return chat()
        else:
                response_content = """# Ready for New Session 🚀

To start a new session, please type:
- **"design"** - Create new UDL assessments
- **"evaluate"** - Analyze existing assessments"""
        
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
            'state': session['state'],
            'branch': session['branch'],
            'message_count': len(session['messages']),
            'context_summary': {
                'has_objectives': bool(session['context']['learning_objectives']),
                'has_grade_level': bool(session['context']['grade_level']),
                'has_subject': bool(session['context']['subject']),
                'has_assessment': bool(session['context']['assessment_content'])
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
        "state": "Start",
        "branch": None,
        "context": {
            "learning_objectives": None,
            "grade_level": None,
            "subject": None,
            "assessment_content": None,
            "assessment_alignment": None
        },
        "created_at": datetime.now().isoformat(),
        "last_activity": datetime.now().isoformat()
    }
    
    save_chat_sessions()
    return jsonify({
        'message': 'Session reset successfully',
        'session_token': session_token,
        'state': 'Start'
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

