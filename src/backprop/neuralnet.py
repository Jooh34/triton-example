import numpy as np
import torch

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # W = weights 매트릭스 초기화, layers = 네트워크아키텍쳐 저장.
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # 첫번째 레이어부터 루프 시작. 마지막 2레이어 전까지 루프.
        for i in np.arange(0, len(layers) - 2):
            # 표준정규분포에서 MxN 행렬 생성, +1 노드는 bias node
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))  # 각 뉴런의 분산으로 normalizing

        # 마지막 두 레이어는 input은 bias term이 필요하지만 output은 필요하지않다.
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))
    
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))
    
    def sigmoid_deriv(self, x):
        return x * (1 - x)
    
    def fit(self, X, y, epochs=1000, displayUpdate=100):
        # X 에 마지막 열에 1로 채워진 행을 추가 (bias trick)
        X = np.c_[X, np.ones((X.shape[0]))]

        # epoch만큼 루프
        for epoch in np.arange(0, epochs):
            # 각 data point를 순회하면서 학습
            for (x, target) in zip(X, y):
                # 한번의 학습 (feedforward -> backward)
                self.fit_partial(x, target)

            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch+1, loss))

    def fit_partial(self, x, y):
        # 2차원보다 작으면 2차원으로변경 (input이 이미지라서 2차원이다.)
        A = [np.atleast_2d(x)]

        ### FEED FORWARD
        for layer in np.arange(0, len(self.W)):
            # net input과 weight를 dot product. 마지막 행 1은 bias로 계산된다.
            net = A[layer].dot(self.W[layer])

            # net output은 nonlinear activation function을 취해서 구한다.
            out = self.sigmoid(net)

            A.append(out)

        ### BACK PROPAGATION
        # 첫번째 스텝은 예상과 정답의 error를 계산.
        error = A[-1] - y
        
        # 여기서부터는 delta 'D' 를 구하기위해서 chain rule을 적용해야한다.
        # 첫번째 엔트리는 error에 activation 함수의 미분값을 곱해주면 된다.
        D = [error * self.sigmoid_deriv(A[-1])]
        
        # chain rule을 이해하고나면 for루프는 작성하기 쉽다.
        for layer in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # reverse deltas (뒤에서부터 계산했으므로)
        D = D[::-1]

        ### WEIGHT UPDATE PHASE
        for layer in np.arange(0, len(self.W)):
            # A와 곱하는 이유는?
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])
        
    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss

def XOR_test():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork([2,2,1], alpha=0.5)
    nn.fit(X, y, epochs=20000)

    # 학습 완료. 테스트
    for (x, target) in zip(X, y):
        pred = nn.predict(x)[0][0]
        step = 1 if pred > 0.5 else 0
        print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(x, target[0], pred, step))

if __name__ == "__main__":
    XOR_test()