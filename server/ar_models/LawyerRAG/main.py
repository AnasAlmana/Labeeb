from lawyer import Lawyer
lawyer_llm = Lawyer()

question = "ما عقوبة التزوير؟"

response = lawyer_llm.inference(question)
print(response)