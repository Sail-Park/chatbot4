import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class SimpleChatBot:
    # 챗봇 클래스 초기화, 파일 경로를 입력받아 질문과 답변을 로드하고 벡터라이저를 초기화
    def __init__(self, filepath):
        self.questions, self.answers = self.load_data(filepath)
        self.vectorizer = TfidfVectorizer()

    # csv 파일을 로드하고 'Q'와 'A' 열을 각각 질문과 답변 리스트로 변환
    def load_data(self, filepath):
        data = pd.read_csv(filepath)
        questions = data['Q'].tolist()  
        answers = data['A'].tolist()  
        return questions, answers

    # 사용자의 입력 문장을 받아, 질문 리스트와의 거리를 계산하고 가장 가까운 질문의 답변을 반환
    def find_best_answer(self, input_sentence):
        similarities = [self.calc_distance(input_sentence, question) for question in self.questions]
        best_match_index = similarities.index(min(similarities))
        return self.answers[best_match_index]

    # 두 문자열 a, b 간의 최소 편집 거리를 계산하는 함수
    def calc_distance(self, a, b):
        if a == b: return 0
        a_len = len(a)
        b_len = len(b)
        if a == "": return b_len
        if b == "": return a_len

        matrix = [[] for i in range(a_len+1)]
        for i in range(a_len+1):
            matrix[i] = [0 for j in range(b_len+1)]
        
        for i in range(a_len+1):
            matrix[i][0] = i
        for j in range(b_len+1):
            matrix[0][j] = j

        # 최소 편집 거리 계산
        for i in range(1, a_len+1):
            ac = a[i-1]
            for j in range(1, b_len+1):
                bc = b[j-1] 
                cost = 0 if (ac == bc) else 1
                matrix[i][j] = min([
                    matrix[i-1][j] + 1,     # 문자 삭제
                    matrix[i][j-1] + 1,     # 문자 삽입
                    matrix[i-1][j-1] + cost # 문자 대체
                ])
        return matrix[a_len][b_len]

# CSV 파일 경로를 지정
filepath = 'ChatbotData.csv'

# 챗봇 인스턴스를 생성
chatbot = SimpleChatBot(filepath)

# '종료'라는 단어가 입력될 때까지 챗봇과의 대화를 반복
while True:
    input_sentence = input('You: ')
    if input_sentence.lower() == '종료':
        break
    response = chatbot.find_best_answer(input_sentence)
    print('Chatbot:', response)