import os
# Set your custom cache directory
os.environ["HF_HOME"] = "/tmp/hf_cache"       # or use os.environ["HF_DATASETS_CACHE"]
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache/transformers"

# Optional: for datasets if you're using them
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache/datasets"

os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers"
CACHE_DIR = "/tmp/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


from flask import Flask, render_template, request, redirect, url_for, session, jsonify, make_response,flash
from flask_cors import CORS
from xhtml2pdf import pisa
from io import BytesIO
import contextlib
import json
import io
import tempfile
import subprocess
from flask import Flask, render_template, request, redirect, url_for, flash
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from flask_ngrok import run_with_ngrok


# Load deepseek model for code optimization and question generation
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"

device = 0 if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto" if device == 0 else None)


app = Flask(__name__)
app.secret_key = '123456'
CORS(app)

# Hardcoded user
USER_DATA = {
    'name': 'abc',
    'email': 'abc@gmail.com',
    'password': '123',
    'skill_level': 'expert'
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    user = {
        'name': USER_DATA['name'],
        'skill': USER_DATA['skill_level']
    }
    return render_template('dashboard.html', user=user)

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('home'))


# Skill Test
@app.route('/skill-test', methods=['GET', 'POST'])
def take_skill_test():
    questions = [
        # Beginner (5 questions)
        {"id": 1, "text": "What is the purpose of a for loop in Python?", "topic": "Loops", "level": "Beginner","options": ["A) Iterate over a sequence", "B) Store data", "C) Define functions", "D) Handle exceptions"]},
        {"id": 2, "text": "Which of the following is an example of recursion?", "topic": "Recursion", "level": "Beginner","options": ["A) A function calling itself", "B) A for loop", "C) A variable declaration", "D) A conditional statement"]},
        {"id": 3, "text": "What does the 'def' keyword do in Python?", "topic": "Functions", "level": "Beginner","options": ["A) Defines a function", "B) Defines a class", "C) Creates a variable", "D) Imports a module"]},
        {"id": 4, "text": "What is inheritance in OOP?", "topic": "OOP", "level": "Beginner","options": ["A) A class inherits properties from another class", "B) A class is defined inside another class",
                 "C) A class contains no methods", "D) A class can be instantiated multiple times"]},
        {"id": 5, "text": "Which of the following is a primitive data type in Python?", "topic": "Data Types", "level": "Beginner","options": ["A) int", "B) list", "C) dict", "D) set"]},

        # Intermediate (5 questions)
        {"id": 6, "text": "How do you declare an array in Python?", "topic": "Arrays", "level": "Intermediate","options": ["A) array = [1, 2, 3]", "B) array = (1, 2, 3)", "C) array = {1, 2, 3}", "D) array = 1, 2, 3"]},
        {"id": 7, "text": "Which function is used to reverse a string in Python?", "topic": "Strings", "level": "Intermediate","options": ["A) reverse()", "B) reversed()", "C) flip()", "D) reverse_string()"]},
        {"id": 8, "text": "Which algorithm is used to sort a list in ascending order in Python?", "topic": "Sorting", "level": "Intermediate","options": ["A) Quick sort", "B) Bubble sort", "C) Merge sort", "D) All of the above"]},
        {"id": 9, "text": "What does binary search do?", "topic": "Searching", "level": "Intermediate","options": ["A) Finds an item in a sorted list", "B) Sorts a list", "C) Performs addition on numbers", "D) Multiplies two numbers"]},
        {"id": 10, "text": "What is a stack data structure?", "topic": "Stacks", "level": "Intermediate","options": ["A) LIFO (Last In, First Out) principle", "B) FIFO (First In, First Out) principle", 
                 "C) A collection of unordered elements", "D) A linear data structure with random access"]},

        # Expert (5 questions)
        {"id": 11, "text": "What is the main feature of a queue data structure?", "topic": "Queues", "level": "Expert","options": ["A) FIFO (First In, First Out) principle", "B) LIFO (Last In, First Out) principle",
                 "C) Dynamic size", "D) Random access"]},
        {"id": 12, "text": "Which operation does a linked list support?", "topic": "Linked Lists", "level": "Expert","options": ["A) Insertion", "B) Deletion", "C) Traversal", "D) All of the above"]},
        {"id": 13, "text": "What is a binary tree?", "topic": "Trees", "level": "Expert","options": ["A) A tree where each node has at most two children", "B) A tree with only two nodes",
                 "C) A tree with multiple children", "D) A tree with no children"]},
        {"id": 14, "text": "What is a graph in data structures?", "topic": "Graphs", "level": "Expert","options": ["A) A set of vertices connected by edges", "B) A linear data structure",
                 "C) A tree with multiple branches", "D) A sorted list"]},
        {"id": 15, "text": "What is hashing used for in data structures?", "topic": "Hashing", "level": "Expert","options": ["A) Storing data in a fixed-size table", "B) Sorting data", "C) Searching data", "D) Storing data in a sequence"]},
    ]


    if request.method == 'POST':
        mcq_answers = {f"q{i}": request.form.get(f"q{i}") for i in range(1, 16)}
        code_answer_1 = request.form.get("code1", "").strip()
        code_answer_2 = request.form.get("code2", "").strip()

        correct_answers = {
            "1": "a",  # For loop purpose: Iterate over a sequence
            "2": "a",  # Recursion example: A function calling itself
            "3": "a",  # 'def' keyword defines a function
            "4": "a",  # Inheritance: class inherits properties from another
            "5": "a",  # Primitive data type: int
            "6": "a",  # Declare array: array = [1, 2, 3]
            "7": "b",  # Reverse string: reversed()
            "8": "d",  # Sorting algorithms: All of the above
            "9": "a",  # Binary search finds item in sorted list
            "10": "a", # Stack data structure is LIFO
            "11": "a", # Queue data structure is FIFO
            "12": "d", # Linked list supports all operations
            "13": "a", # Binary tree: each node max two children
            "14": "a", # Graph: set of vertices connected by edges
            "15": "a", # Hashing: storing data in fixed-size table
        }


        topics = {
            "1": "Loops", "2": "Recursion", "3": "Functions", "4": "OOP",
            "5": "Data Types", "6": "Arrays", "7": "Strings", "8": "Sorting",
            "9": "Searching", "10": "Stacks", "11": "Queues", "12": "Linked Lists",
            "13": "Trees", "14": "Graphs", "15": "Hashing"
        }

        score = 0
        topic_scores = {}
        for q, ans in mcq_answers.items():
            question_number = q[1:]
            if ans is None:
                continue
            ans = ans.strip().lower()
            correct_ans = correct_answers[question_number].strip().lower()
            correct = ans == correct_ans
            score += correct
            topic = topics[question_number]
            topic_scores[topic] = topic_scores.get(topic, 0) + (1 if correct else 0)

        code_score, code_feedback = evaluate_coding_answers({
            'code1': code_answer_1,
            'code2': code_answer_2
        })

        total_score = score + code_score
        strong = [t for t, s in topic_scores.items() if s == 1]
        weak = [t for t, s in topic_scores.items() if s == 0]
        needs_improvement = [t for t, s in topic_scores.items() if 0 < s < 1]
        book = "Data Structures and Algorithms Made Easy by Narasimha Karumanchi"

        if total_score >= 20 and code_score > 5:
            level = "Expert"
        elif total_score >= 13 and code_score >= 5:
            level = "Intermediate"
        else:
            level = "Beginner"

        report = {
            'score': total_score,
            'mcq_score': score,
            'code_score': code_score,
            'level': level,
            'strong_concepts': strong,
            'weak_concepts': weak,
            'needs_improvement': needs_improvement,
            'suggested_book': book,
            'code_feedback': code_feedback
        }

        session['report'] = report
        return render_template('skill_report.html', report=report)

    return render_template("skill_test.html", questions=questions)


# Reset Test
@app.route('/reset-test')
def reset_test():
    session.pop('report', None)
    return redirect(url_for('take_skill_test'))

# Evaluate coding answers

def evaluate_coding_answers(user_code):
    results = []
    score = 0
    expected_output_1 = "True\nFalse\n"
    try:
        output1 = io.StringIO()
        code1 = user_code['code1'] + "\nprint(prime(7))\nprint(prime(8))"
        with contextlib.redirect_stdout(output1):
            exec(code1, {})
        actual_output1 = output1.getvalue()
        correct1 = actual_output1.strip() == expected_output_1.strip()
        score += 5 if correct1 else 0
        results.append(("Prime Checker", correct1, actual_output1.strip()))
    except Exception as e:
        results.append(("Prime Checker", False, f"Error: {e}"))

    expected_output_2 = "120\n"
    try:
        output2 = io.StringIO()
        code2 = user_code['code2'] + "\nprint(fact(5))"
        with contextlib.redirect_stdout(output2):
            exec(code2, {})
        actual_output2 = output2.getvalue()
        correct2 = actual_output2.strip() == expected_output_2.strip()
        score += 5 if correct2 else 0
        results.append(("Factorial", correct2, actual_output2.strip()))
    except Exception as e:
        results.append(("Factorial", False, f"Error: {e}"))

    return score, results

# Run Code
@app.route('/run_code', methods=['POST'])
def run_code():
    data = request.get_json()
    code = data.get("code", "")
    try:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(code, {})
        return jsonify({"output": output.getvalue()})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"})

def optimize_code(code: str) -> str:
    prompt = f"You are a python expert.Optimize the following Python code for better time and space complexity\n\n{code}:"
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_length=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        optimized = tokenizer.decode(outputs[0], skip_special_tokens=True)
        optimized_code = optimized[len(prompt):].strip()
        
        return optimized_code.strip()
    except Exception as e:
        return f"Error optimizing code: {e}"

        
def generate_theory_questions(code: str) -> str:
    prompt = f"Generate theory questions with solutions, based on Data Structures and programming concepts used in the following Python code and not about the input or output:\n\n{code}\n\nQuestions:\n"
    try:
        # Encode input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate output
        outputs = model.generate(
            **inputs,
            max_length=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode full output (prompt + generated)
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from output to get only generated questions
        questions = full_output[len(prompt):].strip()

        return questions

    except Exception as e:
        return f"Error generating questions: {e}"


def explain_error(code: str, error_message: str, level: str) -> str:
    prompt = (
        f"Explain the following Python error for this code snippet and also provide correct code:\n\n"
        f"Code:\n{code}\n\n"
        f"Error:\n{error_message}\n\n"
        f"Explanation for a {level} level:\n"
    )
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt part from generated_text to get clean explanation
        message = generated_text[len(prompt):].strip()
        return message

    except Exception as e:
        return f"⚠️ Error explaining the error: {e}"



# Run Code (Skill)
@app.route('/run_code_skill', methods=['POST'])
def run_code_skill():
    data = request.get_json()
    code = data.get('code', '')
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp:
            temp.write(code)
            temp.flush()
            result = subprocess.run(
                ['python', temp.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )
        output = result.stdout if result.stdout else result.stderr
    except Exception as e:
        output = str(e)
    return jsonify({"output": output})

@app.route('/optimize_code', methods=['POST'])
def optimize_code_api():
    data = request.get_json()
    code = data.get('code', '')
    optimized = optimize_code(code)
    return jsonify({'optimized_code': optimized})

@app.route('/generate_questions', methods=['POST'])
def generate_questions_api():
    data = request.get_json()
    code = data.get('code', '')
    questions = generate_theory_questions(code)
    return jsonify({'questions': questions})

@app.route('/explain_error', methods=['POST'])
def explain_error_api():
    data = request.get_json()
    code = data.get('code', '')
    error_message = data.get('error', '')
    level=data.get('level','')
    explanation = explain_error(code, error_message,level)
    return jsonify({'explanation': explanation})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)


