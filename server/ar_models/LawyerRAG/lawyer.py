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



class Lawyer():
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
        أجب على السؤال بناءً على السياق الموجود في الأسفل.
        إذا كنت لا تعرف الإجابة، أجب بـ"لا أعرف".
        عند الإجابة على السؤال، اذكر اسم المستند ورقم الصفحة ورقم المادة. قم بالإجابة باللغة العربية فقط. تأكد من عدم الإجابة بأخطاء إملائية.

        إجعل اجابتك بالتنسيق التالي:
        الإجابة: نظام الحكم في المملكة العربية السعودية هو نظام ملكي وفقا للمادة الخامسة من النظام الأساسي للحكم رقم (م/11)، الصادر بتاريخ 18/2/1435هـ.
        المستند: النظام الأساسي للحكم
        الصفحة: 0
        المادة: الخامسة

        الإجابة: الحكم في المملكة العربية السعودية يستمد سلطته من كتاب الله تعالى وسنة رسوله، وهما الحاكمان على النظام وجميع أنظمة الدولة.

        المستند: النظام الأساسي للحكم
        الصفحة: 0
        المادة: السابعة

        الإجابة: نعم, يعتبر صناعة خاتم مقلد تزوير وفقا للمادة الثانية من النظام الجزائي لجرائم التزوير. 

        المستند: الانظمة السعودية/النظام الجزائي لجرائم التزوير


        السياق: {context}

        السؤال: {question}
        """

        self.prompt = PromptTemplate.from_template(self.template)

        self.pdf_files = os.listdir("ar_models/LawyerRAG/مجموعة-الانظمة-السعودية")

        self.pages = []
        for pdf_file in self.pdf_files:
            loader = PyPDFLoader("ar_models/LawyerRAG/مجموعة-الانظمة-السعودية/"+pdf_file)
            self.pages.extend(loader.load_and_split())

        vectorstore = DocArrayInMemorySearch.from_documents(self.pages, embedding=self.embeddings)
        self.retriever = vectorstore.as_retriever()



    def inference(self, question):

        

        chain = (
            {
                "context": itemgetter("question") | self.retriever,
                "question": itemgetter("question"),
            }
            | self.prompt
            | self.model
            | self.parser
        )

        
        #questions_responses = {}
        #for question in questions:
        #    questions_responses[question] = chain.invoke({'question': question})
        response = chain.invoke({'question': question})
        return response
    




if __name__ == '__main__':
    lawyer = Lawyer()
    questions = [
                "ما نظام الحكم في المملكة العربية السعودية؟",
                "من اين يستمد الحكم في المملكة العربية السعودية؟",
                "قام شخص بصناعة خاتم مقلد , هل يعتبر هذا تزوير؟",
            ]
    for question in questions:
        print(lawyer.inference(question))
        
