class WordSequence:
    def __init__(self):
        pass
    @staticmethod
    def execute():
        import tensorflow as tf
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import font_manager, rc
        # word2vec (단어의 vector화)
        rc('font',family=font_manager.FontProperties(fname='c:/Windows/Fonts/malgun.ttf').get_name())
        # 단어벡터를 활용해 분석해볼 임의의 문장들
        sentences = ["나 고양이 좋다",
                     "나 강아지 좋다",
                     "나 동물 좋다",
                     "강아지 고양이 동물",
                     "여자친구 고양이 강아지 좋다",
                     "고양이 생선 우유 좋다",
                     "강아지 생선 싫다 우유 좋다",
                     "강아지 고양이 눈 좋다",
                     "나 여자친구 좋다",
                     "여자친구 나 싫다",
                     "여자친구 나 영화 책 음악 좋다",
                     "나 게임 만화 애니 좋다",
                     "고양이 강아지 싫다",
                     "강아지 고양이 좋다"]
        word_sentence =" ".join(sentences).split()
        # print(word_sentence)
        word_list = word_sentence
        word_dict = {W: i for i, W in enumerate(word_list)}
        skip_grams = []
        for i in range(1, len(word_sentence)-1):
            target = word_dict[word_sentence[i]]
            context = [word_dict[word_sentence[i - 1]],
                       word_dict[word_sentence[i + 1]]]
            # (contextm target)
            # 스킵그램을 만든 후, 저장은 단어의 고유번호(index)로 한다
            for W in context:
                skip_grams.append([target,W])
            # ((target, context[0]),(target, context[1]),(target, context[2]))....

        def random_batch(data, size):
            random_inputs = []
            random_labels = []
            random_index = np.random.choice(range(len(data)), size, replace=False)
            # replace = False 면 한번 출력하면 다시 뽑히지 않음
            for i in random_index:
                random_inputs.append(data[i][0]) # target
                random_labels.append([data[i][1]]) #

            return random_inputs, random_labels

        # 러닝 옵션
        training_epoch = 300
        learning_rate =0.1
        batch_size = 20 # 한번에 학습할 데이터 크기
        embeding_size = 2 # 단어 벡터를 구성할 임베딩의 크기
        num_sampled = 15 # 모델 학습에 사용할 샘플의 크기, batch_size 보다는 작아야 함
        voc_size = len(word_list) #총단어의 갯수

        # 모델링
        inputs = tf.compat.v1.placeholder(tf.int32, shape=[batch_size])
        labels = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, 1])
        # tf.nn.ncd_loss 를 사용하려면 출력값을 이렇게 [batch_size, 1] 로 구성해야하 함
        print('embeding_size {}'.format((embeding_size)))
        print('voc_size {}'.format(voc_size))
        embedings = tf.compat.v1.Variable(tf.random.uniform([voc_size, embeding_size], -1.0, 1.0))
        # word2vec 모델의 결과 값만 임베딩 벡터를 저장할 변수
        # 총 단어의 갯수와 임베딩 갯수를 크기로 하는 두개의 차원을 갖습니다
        # embedding vector 의 차원에서 학습할 입력값에 대한 행들을 뽑아옵니다
        '''
        embeddings  input   selected
        [[1,2,3], -> [2,3] ->[[2,3,4],[3,4,5]]
         [2,3,4],
         [3,4,5],
         [4,5,6]
         ]
        '''
        selected_embed = tf.nn.embedding_lookup(embedings, inputs)
        nce_weight = tf.compat.v1.Variable(tf.random.uniform([voc_size,embeding_size], -1.0,1.0))
        nce_bias = tf.compat.v1.Variable(tf.zeros([voc_size]))
        loss = tf.reduce_mean(
            tf.nn.nce_loss(nce_weight, nce_bias,labels, selected_embed, num_sampled, voc_size)
        )
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        # Learning

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            for step in range(1, training_epoch + 1):
                batch_inputs, batch_labels = random_batch(skip_grams, batch_size)
                _, loss_val = sess.run([train_op,loss],
                                      {inputs:batch_inputs, labels:batch_labels})
                if step % 10 ==0:
                    print('loss at step',step," : ",loss_val)
            trained_embeddings =embedings.eval()
            # with 구문안에서 sess.run 대신에 간단히 eval() 함수를 사용해서 저장할 수 있음
        # 테스트
        for i, label in enumerate(word_list):
            x, y =trained_embeddings[i]
            plt.scatter(x,y)
            plt.annotate(label, xy=(x,y), xytext=(5,2),
                         textcoords='offset points',ha ='right', va ='bottom')
        plt.show()