import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter



class Tourism():
    def __init__(self, openai_key = "OPENAI_API_KEY", MODEL="gpt-4o"):
        load_dotenv()
        self.openai_key = openai_key
        self.__OPENAI_API_KEY = os.getenv(self.openai_key)
        self.MODEL = MODEL

        self.parser = StrOutputParser()


        if self.MODEL.startswith("gpt"):
            self.model = ChatOpenAI(api_key=self.__OPENAI_API_KEY, model=self.MODEL)
            self.embeddings = OpenAIEmbeddings()
        else:
            self.model = Ollama(model=self.MODEL)
            self.embeddings = OllamaEmbeddings(model=self.MODEL)

        
        self.template = """
        أنت نموذج من نظام لبيب. مهمتك هي الإجابة على الأسئلة المتعلقة بالسياحة في المملكة العربية السعودية بناءً على المعلومات المتوفرة في الوثائق المحددة.

        احرص على تنوع الترفيه والاماكن الترفيهية،
        إذا كنت لا تعرف الإجابة، أجب بـ "لا أعرف".

        عند الإجابة على السؤال، يرجى الإشارة إلى الأماكن الترفيهية أو المطاعم المتعلقة بالسؤال بالتفصيل. تأكد من الكتابة باللغة العربية الفصحى وتجنب تكرار المطاعم او الاماكن الترفيهية وتجنب الأخطاء الإملائية.

        المعلومات المتاحة: {documents}

        السؤال: {question}
        """


        self.prompt = PromptTemplate.from_template(self.template)



    def inference(self, questions):

        
        pdf_files = os.listdir("ar_models/TourismRAG/pdf_data")

        pages = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader("ar_models/TourismRAG/pdf_data"+pdf_file)
            pages.extend(loader.load_and_split())

        vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=self.embeddings)
        retriever = vectorstore.as_retriever()

        chain = (
            {
                "documents": itemgetter("question") | retriever,
                "question": itemgetter("question"),
            }
            | self.prompt
            | self.model
            | self.parser
        )

        
        questions_responses = {}
        for question in questions:
            questions_responses[question] = chain.invoke({'question': question})
        
        return questions_responses
    




if __name__ == '__main__':
    tourism = Tourism()
    questions = [
                "انا زائر الرياض لمدة ٣ ايام اعطني خطة سياحية ",
            ]

    q_responses = tourism.inference(questions)
    for question, answer in q_responses.items():
        #print("السؤال: ", question)
        print(answer)
