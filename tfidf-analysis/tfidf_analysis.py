import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Đường dẫn tới các tập tin
file_paths = [
    'path/to/Chương trình ĐH.txt',
    'path/to/Chương trình liên thông ĐH.txt',
    'path/to/Chương trình quốc tế.txt',
    'path/to/Chương trình sau ĐH.txt'
]

# Đọc nội dung từ các tập tin
file_contents = []
for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as file:
        file_contents.append(file.read())

# Khởi tạo TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10)

# Áp dụng TF-IDF cho nội dung của các tập tin
tfidf_matrix = tfidf_vectorizer.fit_transform(file_contents)

# Lấy danh sách từ khóa (features)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Lấy điểm số TF-IDF cho từng tập tin
tfidf_scores = tfidf_matrix.toarray()

# Hiển thị từ khóa và điểm TF-IDF cho từng tập tin
for i, file_path in enumerate(file_paths):
    print(f"Top 10 từ khóa cho tập tin {file_path}:")
    for j in range(len(feature_names)):
        print(f"{feature_names[j]}: {tfidf_scores[i][j]}")
    print("\n")
