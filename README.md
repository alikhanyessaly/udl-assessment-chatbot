# UDL Assessment Chatbot

An AI-powered chatbot designed to help K-12 teachers create more inclusive assessment practices using Universal Design for Learning (UDL) principles.

## Features

### Core Functionality
- **Assessment Evaluation**: Analyze existing assessments against UDL principles
- **Assessment Design Support**: Generate multiple UDL-aligned assessment options
- **Token-based Chat History**: No login required, sessions managed via unique tokens
- **UDL Framework Integration**: Built-in knowledge of UDL principles and guidelines

### Key Design Principles
- **Transparency**: Explains UDL reasoning behind each suggestion
- **Teacher Agency**: Presents options rather than prescriptive solutions
- **Cultural Responsiveness**: Encourages consideration of cultural context
- **Professional Development Focus**: Builds teacher capacity for inclusive design

## Setup Instructions

### Prerequisites
- Python 3.7+
- OpenAI API key

### Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp env.example .env
   ```
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://localhost:5000`

## Usage

### Assessment Evaluation
Share an existing assessment description and the chatbot will:
- Identify UDL-aligned strengths
- Flag accessibility and inclusion barriers
- Provide specific improvement suggestions with UDL rationale
- Explain which UDL principles are most relevant

### Assessment Design Support
Describe your learning objectives, grade level, and subject to receive:
- Multiple assessment format options
- Implementation guidance for each option
- UDL rationale explaining inclusive benefits
- Adaptation considerations for your context

### Example Prompts
- "Evaluate my assessment: Students will write a 5-page essay about World War II"
- "Help me design assessment options for: Understanding photosynthesis, Grade 7, Science"
- "What are the key UDL principles for assessment design?"

## UDL Principles Supported

### Principle 1: Multiple Means of Engagement
- Options for recruiting interest
- Options for sustaining effort and persistence
- Options for self-regulation

### Principle 2: Multiple Means of Representation
- Options for perception
- Options for language and symbols
- Options for comprehension

### Principle 3: Multiple Means of Action and Expression
- Options for physical action
- Options for expression and communication
- Options for executive functions

## API Endpoints

- `POST /api/chat` - Main chat endpoint
- `GET /api/session/<token>` - Get chat session history
- `GET /api/sessions` - List all active sessions
- `GET /api/health` - Health check

## Technical Details

- **Backend**: Flask with OpenAI GPT-3.5-turbo integration
- **Frontend**: Modern HTML5/CSS3/JavaScript with responsive design
- **Session Management**: In-memory token-based sessions
- **CORS**: Enabled for cross-origin requests

## Contributing

This tool is designed to augment teacher expertise by building capacity for inclusive design. Suggestions for improvements are welcome, particularly around:
- UDL framework accuracy
- Assessment design examples
- Cultural responsiveness features
- Accessibility improvements

## License

This project is designed for educational use in K-12 public schools in the USA.

