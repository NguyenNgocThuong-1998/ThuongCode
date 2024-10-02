import os
import gensim
from gensim.models import Word2Vec
import re

# Hàm để làm sạch văn bản và tách từ
def preprocess_text(text):
    # Xóa các ký tự không mong muốn và chuyển đổi thành chữ thường
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Tách từ
    tokens = text.split()
    return tokens

# Đọc nội dung từ các tập tin
file_paths = [
    'path/to/Chương trình ĐH.txt',
    'path/to/Chương trình liên thông ĐH.txt',
    'path/to/Chương trình quốc tế.txt',
    'path/to/Chương trình sau ĐH.txt'
]

file_contents = []
for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # Tiền xử lý văn bản
        tokens = preprocess_text(content)
        file_contents.append(tokens)

# Tạo mô hình Word2Vec với các từ tách được từ văn bản
word2vec_model = Word2Vec(sentences=file_contents, vector_size=100, window=5, min_count=2, workers=4)

# Huấn luyện mô hình
word2vec_model.train(file_contents, total_examples=len(file_contents), epochs=10)

# Kiểm tra những từ có ngữ nghĩa gần nhất với một từ nào đó
similar_words = word2vec_model.wv.most_similar("học", topn=5)
print("Những từ có ngữ nghĩa gần nhất với 'học':")
for word, score in similar_words:
    print(f"{word}: {score}")

# Tìm từ khác có quan hệ với "sinh" và "viên"
similar_words_combined = word2vec_model.wv.most_similar(positive=['sinh', 'viên'], topn=5)
print("\nNhững từ có quan hệ gần với cả 'sinh' và 'viên':")
for word, score in similar_words_combined:
    print(f"{word}: {score}")
