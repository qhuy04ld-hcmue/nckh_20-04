import os
import pickle
import io
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from google.cloud import vision
from googleapiclient.discovery import build

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

YOUTUBE_API_KEY = ""#thay vao nha
JSON_API_PATH = "./api_gg.json"  # Cập nhật đường dẫn API key JSON

# Load mô hình phân loại
with open("text_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Dự đoán môn học từ văn bản đầu vào
def predict_text(text):
    return model.predict([text])[0]

# Tìm kiếm video trên YouTube
def get_youtube_videos(query, max_results=20):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    try:
        request = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=max_results
        )
        response = request.execute()
        videos = []
        for item in response.get("items", []):
            video_data = {
                "title": item["snippet"]["title"],
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"]
            }
            videos.append(video_data)
        
        # Nếu không tìm thấy video, thử tìm với từ khóa chung hơn
        if not videos:
            simplified_query = "học Lý Hóa online"
            request = youtube.search().list(
                q=simplified_query,
                part="snippet",
                type="video",
                maxResults=max_results
            )
            response = request.execute()
            for item in response.get("items", []):
                video_data = {
                    "title": item["snippet"]["title"],
                    "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                    "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"]
                }
                videos.append(video_data)
        
        return videos
    except Exception as e:
        print("Lỗi API YouTube:", e)
        return []

# Trích xuất văn bản từ ảnh
def extract_text_from_image(file, json_api_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = json_api_path
    client = vision.ImageAnnotatorClient()
    content = file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if response.error.message:
        raise Exception(f"Google Vision API Error: {response.error.message}")
    return texts[0].description if texts else ""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    videos = []
    extracted_text = None

    if request.method == 'POST':
        user_text = request.form.get('text', '')

        # Kiểm tra nếu có file ảnh được tải lên
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                extracted_text = extract_text_from_image(file, JSON_API_PATH)
                user_text += " " + extracted_text  # Gộp nội dung nhập tay + ảnh

        if user_text:
            prediction = predict_text(user_text)
            print("Dự đoán môn học:", prediction)
            search_query = f"{user_text} {prediction} học online"
            videos = get_youtube_videos(search_query, max_results=20)

        return render_template('index.html', extracted_text=extracted_text, prediction=prediction, videos=videos)

    return render_template('index.html', extracted_text=None, prediction=None, videos=[])

if __name__ == '__main__':
    app.run(debug=True)




""" code nay ngon va on nek
import os
import pickle
import io
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from google.cloud import vision
from googleapiclient.discovery import build

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

YOUTUBE_API_KEY = "AIzaSyDftvUOIo3x5l5LdU_9toameZ31nTKHaz0"

# Load mô hình phân loại
with open("text_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Dự đoán môn học từ văn bản đầu vào
def predict_text(text):
    return model.predict([text])[0]

# Tìm kiếm video trên YouTube
def get_youtube_videos(query, max_results=20):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    try:
        request = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=max_results
        )
        response = request.execute()
        videos = []
        for item in response.get("items", []):
            video_data = {
                "title": item["snippet"]["title"],
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"]
            }
            videos.append(video_data)
        return videos
    except Exception as e:
        print("Lỗi API YouTube:", e)
        return []

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    videos = []
    extracted_text = None  # Lưu văn bản từ file (nếu có)

    if request.method == 'POST':
        user_text = request.form.get('text', '')

        # Nếu có file được tải lên, trích xuất văn bản từ file
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                client = vision.ImageAnnotatorClient()
                content = file.read()
                image = vision.Image(content=content)
                response = client.text_detection(image=image)
                texts = response.text_annotations
                if texts:
                    extracted_text = texts[0].description  # Lấy toàn bộ văn bản
                    user_text += " " + extracted_text  # Kết hợp với nội dung nhập vào

        if user_text:
            prediction = predict_text(user_text)
            print("Dự đoán môn học:", prediction)  # Debugging
            search_query = user_text  # Dùng toàn bộ input làm từ khóa tìm kiếm
            videos = get_youtube_videos(search_query, max_results=20)

        return render_template('index.html', extracted_text=extracted_text, prediction=prediction, videos=videos)

    return render_template('index.html', extracted_text=None, prediction=None, videos=[])

if __name__ == '__main__':
    app.run(debug=True)
"""


""" 22-03
import os
import pickle
import io
import re
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from google.cloud import vision
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

YOUTUBE_API_KEY = "AIzaSyDftvUOIo3x5l5LdU_9toameZ31nTKHaz0"

# Load mô hình phân loại
with open("text_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Cập nhật cơ sở dữ liệu khóa học với mô tả chi tiết
course_database = [
    {"title": "Vật lý cơ bản", "url": "https://www.example.com/ly-co-ban", "description": "Khóa học Vật lý cơ bản với các chủ đề như động lực học, lực và mô men."},
    {"title": "Vật lý nâng cao", "url": "https://www.example.com/ly-nang-cao", "description": "Khóa học nâng cao về dao động, quang học và các lĩnh vực chuyên sâu của vật lý."},
    {"title": "Hóa học cơ bản", "url": "https://www.example.com/hoa-co-ban", "description": "Khóa học về axit, bazo, hóa trị, với các thí nghiệm cơ bản trong Hóa học."},
    {"title": "Hóa học nâng cao", "url": "https://www.example.com/hoa-nang-cao", "description": "Khóa học nâng cao về phản ứng hóa học, muối và điện hóa."},
]

# Dự đoán môn học từ văn bản đầu vào
def predict_text(text):
    return model.predict([text])[0]

# Tính toán độ tương đồng giữa văn bản đầu vào và mô tả khóa học sử dụng TF-IDF và Cosine Similarity
def find_best_matching_courses(text):
    # Tạo vector TF-IDF từ mô tả khóa học
    course_descriptions = [course["description"] for course in course_database]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(course_descriptions + [text])
    
    # Tính cosine similarity giữa văn bản và các mô tả khóa học
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    # Sắp xếp và lấy các khóa học có độ tương đồng cao nhất
    similarity_scores = cosine_similarities.flatten()
    sorted_course_indices = similarity_scores.argsort()[::-1]
    matched_courses = [course_database[i] for i in sorted_course_indices if similarity_scores[i] > 0.1]
    
    return matched_courses

# Trích xuất văn bản từ hình ảnh sử dụng Google Vision API
def extract_text_from_image(image_path, json_api_path):
    # Cấu hình đường dẫn tới API Google Cloud Vision
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = json_api_path
    client = vision.ImageAnnotatorClient()
    
    # Đọc nội dung hình ảnh
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    # Chuyển hình ảnh thành đối tượng Google Vision Image
    image = vision.Image(content=content)
    
    # Gửi yêu cầu nhận diện văn bản từ hình ảnh
    response = client.text_detection(image=image)
    
    # Lấy kết quả nhận diện văn bản
    texts = response.text_annotations
    if response.error.message:
        raise Exception(f"Google Vision API Error: {response.error.message}")
    
    # Trả về văn bản đầu tiên nếu có
    return texts[0].description if texts else ""

# Lọc khóa học theo từ khóa (có thể giữ lại nếu bạn muốn kết hợp với các phương pháp khác)
def filter_courses_by_keywords(text):
    text = text.lower()
    matched_courses = []
    for course in course_database:
        if any(keyword in text for keyword in course["description"].split()):
            matched_courses.append(course)
    return matched_courses

# Tìm kiếm video trên YouTube
def get_youtube_videos(query, max_results=12):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    try:
        request = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=max_results
        )
        response = request.execute()
        videos = []
        for item in response.get("items", []):
            video_data = {
                "title": item["snippet"]["title"],
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"]
            }
            videos.append(video_data)
        return videos
    except Exception as e:
        print("Lỗi API YouTube:", e)
        return []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        extracted_text = None
        prediction = None
        courses = []
        videos = []

        if 'text' in request.form and request.form['text']:
            user_text = request.form['text']
            prediction = predict_text(user_text)
            # Tìm khóa học dựa trên độ tương đồng văn bản
            courses = find_best_matching_courses(user_text)
            search_query = user_text
            videos = get_youtube_videos(search_query)

        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.txt'):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(file_path)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                prediction = predict_text(text)
                courses = find_best_matching_courses(text)
                search_query = text
                videos = get_youtube_videos(search_query)
            elif file.filename.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(file_path)
                extracted_text = extract_text_from_image(file_path, './api_gg.json')
                prediction = predict_text(extracted_text)
                courses = find_best_matching_courses(extracted_text)
                search_query = extracted_text
                videos = get_youtube_videos(search_query)

        return render_template('index.html', extracted_text=extracted_text, prediction=prediction, courses=courses, videos=videos)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
"""



'''
import os
import pickle
import io
import re
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from google.cloud import vision
from googleapiclient.discovery import build

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

YOUTUBE_API_KEY = "AIzaSyDftvUOIo3x5l5LdU_9toameZ31nTKHaz0"

# Load mô hình phân loại
with open("text_classifier.pkl", "rb") as f:
    model = pickle.load(f)

course_database = [
    {"title": "Vật lý cơ bản", "url": "https://www.example.com/ly-co-ban", "keywords": ["vật lý", "lực", "động lực học"]},
    {"title": "Vật lý nâng cao", "url": "https://www.example.com/ly-nang-cao", "keywords": ["dao động", "quang học", "cơ học"]},
    {"title": "Hóa học cơ bản", "url": "https://www.example.com/hoa-co-ban", "keywords": ["axit", "bazo", "hóa trị"]},
    {"title": "Hóa học nâng cao", "url": "https://www.example.com/hoa-nang-cao", "keywords": ["phản ứng", "muối", "điện hóa"]},
]

def predict_text(text):
    return model.predict([text])[0]

def extract_text_from_image(image_path, json_api_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = json_api_path
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if response.error.message:
        raise Exception(f"Google Vision API Error: {response.error.message}")
    return texts[0].description if texts else ""

def filter_courses_by_keywords(text):
    text = text.lower()
    matched_courses = []
    for course in course_database:
        if any(keyword in text for keyword in course["keywords"]):
            matched_courses.append(course)
    return matched_courses

def refine_search_query(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = text.split()
    keywords = " ".join(words[:4])
    return keywords if keywords else "khóa học khoa học"

def get_youtube_videos(query, max_results=12):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    try:
        request = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=max_results
        )
        response = request.execute()
        videos = []
        for item in response.get("items", []):
            video_data = {
                "title": item["snippet"]["title"],
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"]
            }
            videos.append(video_data)
        return videos
    except Exception as e:
        print("Lỗi API YouTube:", e)
        return []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        extracted_text = None
        prediction = None
        courses = []
        videos = []

        if 'text' in request.form and request.form['text']:
            user_text = request.form['text']
            prediction = predict_text(user_text)
            courses = filter_courses_by_keywords(user_text)
            search_query = refine_search_query(user_text)
            videos = get_youtube_videos(search_query)

        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.txt'):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(file_path)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                prediction = predict_text(text)
                courses = filter_courses_by_keywords(text)
                search_query = refine_search_query(text)
                videos = get_youtube_videos(search_query)
            elif file.filename.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(file_path)
                extracted_text = extract_text_from_image(file_path, './api_gg.json')
                prediction = predict_text(extracted_text)
                courses = filter_courses_by_keywords(extracted_text)
                search_query = refine_search_query(extracted_text)
                videos = get_youtube_videos(search_query)

        return render_template('index.html', extracted_text=extracted_text, prediction=prediction, courses=courses, videos=videos)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
'''