'''
 포유동물 모델링 튜토리얼 ([b,b] -> 3가지 분류)
 # [털, 날개] -> 기타, 포유류, 조류
 [1,0], -> [0,1,0] 포유류
 [1,1], -> [0,0,1] 조류
 [0,0], -> [1,0,0] 기타
 [0,1], -> [0,0,1] 조류
 '''

class Mammal:
    def __init__(self):
        pass
    @staticmethod
    def execute():
        import tensorflow as tf
        import numpy as np

        x_data = np.array(
            [[0,0],
             [1,0],
             [1,1],
             [0,0],
             [0,0],
             [0,1],]
        )
        y_data = np.array(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1],
             [1, 0, 0],
             [1, 0, 0],
             [0, 0, 1]
             ]
        )
        print('총 데이터의 갯수 {}개'.format(len(x_data)))
        print('x_shape = {} #털이있거나 날개가있거나 [b,b] 형태로 input\ny_shape = {} #3가지로 분류하며 해당값에 1로 return 하며 tf.argmax(Y, 1)로 index값 반환'.format(x_data.shape,y_data.shape))
        X = tf.compat.v1.placeholder(tf.float32)
        Y = tf.compat.v1.placeholder(tf.float32)
        W = tf.Variable(tf.random.uniform([2,3],-1,1.))
        # -1은 all
        # 신경망 neural network 앞으로는 nn으로 표기 요즘은 nn사용
        # nn은 2차원으로 [입력층(특성), 출력층(레이블)] ->[2,3] 으로 정합니다
        b = tf.Variable(tf.zeros([3]))
        # b는 각레이어의 아웃풋 갯수로 설정, 최종 결과값의 분류 갯수 3설정
        L = tf.add(tf.matmul(X,W),b) # L = W x X + b
        L = tf.nn.relu(L)
        model = tf.nn.softmax(L)
        # softmax 소프트맥스 함수는 전체 결과값의 합을 1로 만들어 확률로 나타냄
        print('--- 모델내부 보기 ---')
        print(model)
        cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.math.log(model),axis=1))
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(cost)
        # 비용함수를 최소화 시키면 (=경사도를 0으로 만들면) 그 값이 최적화 된 값이다.
        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)
        횟수 = 250
        for step in range(횟수):
            sess.run(train_op,{X:x_data,Y:y_data})
            if (step + 1)%10 ==0:
                print(step+1, sess.run(cost,{X:x_data,Y:y_data}))

        # 결과확인

        prediction = tf.argmax(model, 1)
        target = tf.argmax(Y,1)
        print(x_data)
        print('예측값:',sess.run(model,{X:x_data}))
        print('실제값:',sess.run(target,{Y:y_data}))
        # tf.argmax : 예측값과 실제값의 행렬에서 tf.arg_max 를 이용해 가장 큰 값을 가져옴
        # ex) [[0,1,1][1,0,0]] - > [1,0]
        # [[0.2,0.7,0.1][0.9,0.1,0]] - > [1,0]
        is_correct = tf.equal(prediction, target)
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        print('정확도 : %.2f'%sess.run(accuracy*100,{X:x_data,Y:y_data}))
        # 데이터 양의 부족으로 일정횟수이상부터는 똑같음