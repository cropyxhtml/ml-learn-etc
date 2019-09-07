from n04_tensorflow_test.mammal import Mammal
from n04_tensorflow_test.word_sequence import WordSequence
from n04_tensorflow_test.web_crawler import WebCrawler
from n04_tensorflow_test.naive_bayes import NaiveBayesClassfier
from n04_tensorflow_test.mail_checker_ctrl import MailCheckerController


if __name__ == '__main__':
    # t = Mammal()
    # Mammal.execute() #1
    # WordSequence.execute() #2

    # t = WebCrawler.create_model() #3
    # nb =NaiveBayesClassfier
    # nb.train('./data/','review_train.csv')
    # print(nb.classify('내 인생에서 최고의 영화'))

    ctrl = MailCheckerController()
    ctrl.run()