import time
import numpy as np
import os
import math
import itertools
from CEC13 import CEC13

EPS = 1e-1
import pandas as pd
import matplotlib.pyplot as plt


class BALL:
    def __init__(self, center, radius, value, dim):
        self.center = center
        self.radius = radius
        self.value = value
        self.dim = dim



class GBO:
    def __init__(self):
        # 参数

        # 方法的参数
        self.radius_rate = None  # 半径衰减比,函数传入当前迭代，和最大迭代，返回半径衰减比例
        self.init_size = None  # 初始粒球个数
        self.ball_size = None  # 过程球的个数
        self.radius_eps = None  # 半径阈值
        self.inps_size = None  # 内部采点的个数

        # 问题的参数
        self.evaluator = None  # 适应度计算函数
        self.dim = None
        self.upper_bound = None
        self.lower_bound = None
        self.fes = None  # 迭代
        self.max_fes = None

    def load_prob(self,
                  # 问题的参数
                  evaluator=lambda x: np.sin(x),
                  dim=1,
                  upper_bound=100,
                  lower_bound=-100,
                  fes=-1,
                  # 方法的参数
                  radius_rate=0.6,
                  # radius_rate=lambda cur_fes, m_fes: 0.5,
                  init_size=10,
                  radius_eps=1e-1,
                  inps_size=20):
        # 载入参数
        self.evaluator = evaluator
        self.dim = dim
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.fes = fes
        self.max_fes = dim * 10000  # 非传入参数

        self.radius_rate = radius_rate
        self.init_size = init_size
        self.ball_size = init_size
        self.radius_eps = radius_eps
        self.inps_size = inps_size
        # 初始化随机种子
        np.random.seed(int(os.getpid() * time.perf_counter() / 10000))

    def run(self):
        # running time
        begin_time = time.perf_counter()
        # init balls

        balls = []
        recycle_balls = []
        size_record = [self.init_size]

        centers = np.random.uniform(self.lower_bound,
                                    self.upper_bound,
                                    (self.init_size, self.dim))
        radius = np.random.uniform(50, 70, (self.init_size,))
        fits = self.fit_compute(centers)

        all_idvs = [centers]
        all_fits = [fits]

        for i in range(self.init_size):
            # 计算球的边界点及其适应度值和梯度
            temp_center = centers[i, :]
            temp_radius = radius[i]
            balls.append(BALL(temp_center, temp_radius, fits[i], self.dim))

        fits = np.concatenate(all_fits, axis=0)
        idvs = np.concatenate(all_idvs, axis=0)
        best_idx = np.argmin(fits)
        best_fit = fits[best_idx]
        best_idv = idvs[best_idx]

        history_best_fit = []
        history_best_idv = []

        history_best_fit.append(best_fit)
        history_best_idv.append(best_idv)

        # 开始主循环
        while True:
            # print("haha")
            # 没计算资源了
            if self.fes > self.max_fes:
                break
            # 记录
            all_fits = []
            all_idvs = []
            n_balls = []

            # 处理球
            for ball in balls:

                # 回收半径比较小的球
                if ball.radius < self.radius_eps:
                    recycle_balls.append(ball)
                    continue

                # bias = np.random.uniform(-1, 1, [max(int(self.inps_size*ball.radius/50),1), self.dim])
                bias = np.random.uniform(-1, 1, [self.inps_size, self.dim])
                inps = ball.center + bias * ball.radius

                inps = self._map(inps,  ball.center, ball.radius)
                inps_value = self.fit_compute(inps)

                all_idvs.append(inps)
                all_fits.append(inps_value)

                min_idx = np.argmin(inps_value)
                min_value = inps_value[min_idx]

                # 如果有其他最小点,
                if min_value < ball.value:
                    temp_center = inps[min_idx]
                    # temp_radius = ball.radius * self.radius_rate
                    temp_radius = ball.radius
                    n_balls.append(BALL(temp_center, temp_radius, min_value, self.dim))

                # temp_center = inps[min_idx]
                # # temp_radius = ball.radius * self.radius_rate
                # temp_radius = ball.radius
                # n_balls.append(BALL(temp_center, temp_radius, min_value, self.dim))

                temp_center = ball.center
                temp_radius = ball.radius * self.radius_rate
                n_balls.append(BALL(temp_center, temp_radius, ball.value, self.dim))

            fits = np.concatenate(all_fits, axis=0)
            idvs = np.concatenate(all_idvs, axis=0)
            # 记录更新全局最优
            best_idx = np.argmin(fits)
            if best_fit > fits[best_idx]:
                best_fit = fits[best_idx]
                best_idv = idvs[best_idx]

            history_best_fit.append(best_fit)
            history_best_idv.append(best_idv)

            # 更新球集合和当前个数
            # balls = n_balls
            balls = sorted(n_balls, key=lambda ball: ball.value)[:min(len(n_balls), 3*self.init_size)]
            self.ball_size = len(balls)
            size_record.append(self.ball_size)
        run_time = time.perf_counter() - begin_time
        return best_fit, run_time, best_idv, history_best_fit, size_record, len(recycle_balls)

    def fit_compute(self, dps):
        # 基于evaluator的矩阵计算适应度值
        # Args:
        #   dps(np二维矩阵)：要计算适应度值的边界点合集
        # Returns:
        #   fitness(np一维矩阵)：对应点的适应度值
        pshape = np.shape(dps)
        fitness = np.zeros(pshape[0])
        for i in range(pshape[0]):
            temp = self.evaluator(dps[i, :])
            fitness[i] = temp
            self.fes += 1
        return fitness

    # def _map(self, samples, ball, ball_radius):
    #     # 将samples中超出边界的点随机映射到以ball为圆心，radius为半径，且不超过边界的位置
    #     # Args:
    #     #   samples(np二维矩阵)：要映射的点
    #     #   ball(np一维矩阵)：中心位置
    #     #   radius(float): 该球对应半径
    #     # Returns:
    #     #   samples(np二维矩阵)：映射后的点
    #     in_bound = (samples > self.lower_bound) * (samples < self.upper_bound)
    #     rand_samples = np.random.uniform(max(ball[0] - ball_radius, self.lower_bound),
    #                                      min(ball[0] + ball_radius, self.upper_bound),
    #                                      (1, len(samples)))
    #     for i in range(1, len(ball)):
    #         temp_samples = np.random.uniform(max(ball[i] - ball_radius, self.lower_bound),
    #                                          min(ball[i] + ball_radius, self.upper_bound),
    #                                          (1, len(samples)))
    #         rand_samples = np.concatenate((rand_samples, temp_samples))
    #     rand_samples = rand_samples.transpose()
    #     samples = in_bound * samples + (1 - in_bound) * rand_samples
    #     return samples

    def _map(self, samples, ball, ball_radius):
        # 将samples中超出边界的点随机映射
        # Args:
        #   samples(np二维矩阵)：要映射的点
        # Returns:
        #   samples(np二维矩阵)：映射后的点
        pshape = np.shape(samples)
        for i in range(pshape[0]):
            if np.any(samples[i] < self.lower_bound):
                bool_matrix = samples[i] < self.lower_bound
                samples[i][np.squeeze(bool_matrix)] = self.lower_bound + samples[i][np.squeeze(bool_matrix)] % (
                        self.upper_bound - self.lower_bound)
            if np.any(samples[i] > self.upper_bound):
                bool_matrix = samples[i] > self.upper_bound
                samples[i][np.squeeze(bool_matrix)] = self.lower_bound + samples[i][np.squeeze(bool_matrix)] % (
                        self.upper_bound - self.lower_bound)
        return samples



fun_num = 26
sum = 0
dim = 30
run_n = 50
function_best = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600, -500, -400, -300, -200, -100, 100, 200, 300,
                 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]
the_mean = []
for j in range(fun_num-1, 28):
    print("fun_num:", j + 1)
    sum = 0
    rosenbrock = lambda x: CEC13(x.reshape((1, dim)), j + 1)
    for i in range(run_n):
        print("RUN:", i)
        gbo = GBO()
        gbo.load_prob(evaluator=rosenbrock, radius_rate=0.7, dim=dim, init_size=5, lower_bound=-100, upper_bound=100,
                      radius_eps=0, inps_size=4 * dim)
        best_fit, run_time, best_idv, history_best_fit, size_record, recycle_balls = gbo.run()
        sum += best_fit - function_best[j]
        print("Best solution:", best_idv)
        # print("Is stop:", is_stop)
        print("Best fitness value:", best_fit - function_best[j])
        # print("The ball's size_record:", size_record)
        print("Running time:", run_time, "seconds")
    the_mean.append(sum / run_n)
    print("The fun_num", j, " mean:", sum / run_n)
    print("\n")

with open("lxy实验/GBGS内部随机采点加球（优化）30D——50-70-保留3*init_size.txt", "w") as file:
    for item in the_mean:
        file.write(str(item) + "\n")
