import time
import numpy as np
import os


class BALL:
    def __init__(self, center, radius, value, dim):
        self.center = center
        self.radius = radius
        self.value = value
        self.dim = dim


class GBO:
    def __init__(self):
        # Parameters

        # Method parameters
        self.radius_rate = None  # Radius decay rate, function takes current iteration and max iteration as input, returns radius decay ratio
        self.ball_size = None  # Number of balls retained per generation
        self.max_iter = None  # Maximum number of iterations

        # Problem parameters
        self.evaluator = None  # Fitness evaluation function
        self.dim = None
        self.upper_bound = None
        self.lower_bound = None
        self.fes = None  # Number of evaluations
        self.max_fes = None

    def load_prob(self,
                      # Problem parameters
                      evaluator=lambda x: np.sum(x ** 2),
                  dim=30,
                  upper_bound=100.0,
                  lower_bound=-100.0,
                  fes=-1,
                  # Method parameters
                  radius_rate=0.96,
                  init_size=1,
                  ball_size=30,
                  max_iter=250,
                  max_fes=300000):
        # Load parameters
        self.evaluator = evaluator
        self.dim = dim
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.fes = fes
        self.max_fes = max_fes

        self.radius_rate = radius_rate
        self.ball_size = ball_size
        self.max_iter = max_iter
        # Initialize random seed
        np.random.seed(int(os.getpid() * time.perf_counter() / 10000))

    def run(self):
        # Running time
        begin_time = time.perf_counter()
        # Initialize balls
        balls = []
        size_record = [1]
        centers = np.array([[(self.upper_bound + self.lower_bound) / 2] * self.dim])
        radius = np.full(1, (self.upper_bound - self.lower_bound) / 2)
        fits = self.fit_compute(centers)

        all_idvs = [centers]
        all_fits = [fits]

        # Compute boundary points of the ball and their fitness values and gradients
        temp_center = centers[0, :]
        temp_radius = radius[0]
        balls.append(BALL(temp_center, temp_radius, fits[0], self.dim))

        fits = np.concatenate(all_fits, axis=0)
        idvs = np.concatenate(all_idvs, axis=0)
        best_idx = np.argmin(fits)
        best_fit = fits[best_idx]
        best_idv = idvs[best_idx]

        history_best_fit = []
        history_best_idv = []

        history_best_fit.append(best_fit)
        history_best_idv.append(best_idv)
        iteration = -1
        # Start main loop
        while True:
            iteration += 1
            if self.fes > self.max_fes:
                break
            # Record
            all_fits = []
            all_idvs = []
            n_balls = []

            guide_num = 2
            bias_num = self.max_fes / len(balls) / self.max_iter - guide_num
            # Process balls
            for ball in balls:
                # Compute sampling-related data
                sub_ball = []
                all_samples = np.random.uniform(-1, 1, [int(bias_num), self.dim])
                all_inps = ball.center + all_samples * ball.radius
                all_inps = self._map(all_inps)
                all_inps_value = self.fit_compute(all_inps)

                all_idvs.append(all_inps)
                all_fits.append(all_inps_value)

                for i in range(len(all_inps)):
                    if not self.is_point_inside_any_ball(all_inps[i], sub_ball):
                        sub_ball.append(
                            BALL(all_inps[i], ball.radius * self.radius_rate, all_inps_value[i], self.dim))

                # Guide ball points
                top_num = int(bias_num * 0.2)
                sort_idx = np.argsort(all_inps_value)
                top_idx = sort_idx[-top_num:]
                btm_idx = sort_idx[:top_num]

                top_mean = np.mean(all_inps[top_idx, :], axis=0)
                btm_mean = np.mean(all_inps[btm_idx, :], axis=0)
                delta = (top_mean - btm_mean)

                w = np.random.uniform(0.5, 1.5, (guide_num, 1))
                guide_inps = btm_mean - delta * w
                guide_inps = self._map(guide_inps)
                guide_fit = self.fit_compute(guide_inps)

                all_idvs.append(guide_inps)
                all_fits.append(guide_fit)

                for i in range(len(guide_inps)):
                    sub_ball.append(BALL(guide_inps[i], ball.radius * self.radius_rate, guide_fit[i], self.dim))

                n_balls += sub_ball

            fits = np.concatenate(all_fits, axis=0)
            idvs = np.concatenate(all_idvs, axis=0)
            # Record and update global optimum
            best_idx = np.argmin(fits)
            if best_fit > fits[best_idx]:
                best_fit = fits[best_idx]
                best_idv = idvs[best_idx]

            history_best_fit.append(best_fit)
            history_best_idv.append(best_idv)
            balls = sorted(n_balls, key=lambda ball: ball.value)[:min(len(n_balls), self.ball_size)]
            size_record.append(len(balls))
        run_time = time.perf_counter() - begin_time
        return best_fit, run_time, best_idv

    def fit_compute(self, dps):
        # Calculate fitness values based on evaluator for a matrix
        # Args:
        #   dps(np 2D array): Set of boundary points to calculate fitness values for
        # Returns:
        #   fitness(np 1D array): Corresponding fitness values for the points
        pshape = np.shape(dps)
        fitness = np.zeros(pshape[0])
        for i in range(pshape[0]):
            temp = self.evaluator(dps[i, :])
            fitness[i] = temp
            self.fes += 1
        return fitness

    def _map(self, samples):
        # Randomly map points in samples that exceed boundaries
        # Args:
        #   samples(np 2D array): Points to be mapped
        # Returns:
        #   samples(np 2D array): Mapped points
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

    def is_point_inside_any_ball(self, point, balls):
        """Check if the point is inside any of the given balls."""
        for ball in balls:
            if np.linalg.norm(point - ball.center) < ball.radius:
                return True
        return False
