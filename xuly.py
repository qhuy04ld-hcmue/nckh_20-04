from vncorenlp import VnCoreNLP
from collections import Counter

# Đường dẫn đến mô hình VnCoreNLP (file .jar)
vncorenlp = VnCoreNLP("D:/vnnlp/master/vncorenlp.jar")

def extract_keywords(text, top_n=5):
    # Phân tích cú pháp văn bản bằng VnCoreNLP
    sentences = vncorenlp.tokenize(text)  # Tokenize văn bản
    words = []

    # Duyệt qua mỗi câu đã được tách từ
    for sentence in sentences:
        for word in sentence:  # Chỉ lấy từ, không cần unpack
            words.append(word)
    
    # Sử dụng Counter để đếm tần suất các từ
    word_freq = Counter(words)
    
    # Lọc các từ phổ biến (ngừng từ)
    stopwords = set(['của', 'và', 'theo', 'này', 'với', 'là', 'trong', 'một', 'để', 'có', 'đã'])
    keywords = [word for word, freq in word_freq.most_common() if word not in stopwords]
    
    return keywords[:top_n]

# Ví dụ văn bản cần xử lý
text = """Viết phương trình hoá học của phản 
ứng xảy ra khi cho nhôm (aluminium) 
và kẽm (zinc) tác dụng với sulfur."""
text_ly = """ Một máy bay nhỏ có khối lượng 690 kg đang chạy trên đường băng để cất cánh với động năng 25.103 J.

a. Tính tốc độ của máy bay.

b. Khi bắt đầu cất cánh, tốc độ máy bay tăng gấp 3 lần giá trị trên. Tính động năng của máy bay khi đó."""

text1 = """
Công nghệ thông tin là lĩnh vực nghiên cứu và phát triển các hệ thống máy tính và phần mềm. 
Trong đó, một số lĩnh vực nổi bật bao gồm trí tuệ nhân tạo, học máy và dữ liệu lớn. 
Các nghiên cứu trong công nghệ thông tin đóng góp rất lớn vào sự phát triển của nền kinh tế toàn cầu.
"""

# Tóm tắt các từ khóa chính
keywords = extract_keywords(text, top_n=5)

# In kết quả
print("Các từ khóa chính:", keywords)
