import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

rng = np.random.default_rng()

MEAS_NOISE_STD = 0.01
UPD_NOISE_STD = 0.1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class TreeNode:
    def __init__(self, x, prob, weight, input):
        self.val = x
        self.prob = prob
        self.weight = weight
        self.input = input
        self.left = None
        self.right = None


def node_operation(weight, input):
    W, b = weight
    b = b.squeeze()
    # out = sigmoid(W*input+b)
    out = sigmoid(np.einsum("i,i", W, input) + b)
    return (out, 1 - out)


def leaf_prediction(weight, input):
    W, b = weight
    b = b.squeeze()
    # out = W*input+b

    out = np.einsum("i,i", W, input) + b
    return out


def buildTree(nums, weights, input, max_depth):
    start_index = 2 * (max_depth)
    final_index = 2 ** (max_depth + 1)
    final_prediction_list = []
    if not nums:
        return None
    root = TreeNode(x=nums[0], prob=1, weight=weights[0], input=input)
    q = [root]
    i = 1
    while i < len(nums):
        curr = q.pop(0)
        left_prob, right_prob = node_operation(curr.weight, curr.input)
        left_prob *= curr.prob
        right_prob *= curr.prob
        if i < len(nums):

            curr.left = TreeNode(x=nums[i], prob=left_prob, weight=weights[i], input=input)
            q.append(curr.left)
            if curr.left.val >= start_index and curr.left.val < final_index:
                pred = leaf_prediction(curr.left.weight, input)
                pred *= left_prob
                final_prediction_list.append(pred)

            i += 1

        if i < len(nums):

            curr.right = TreeNode(x=nums[i], prob=right_prob, weight=weights[i], input=input)
            q.append(curr.right)
            if curr.right.val >= start_index and curr.right.val < final_index:
                pred = leaf_prediction(curr.right.weight, input)
                pred *= right_prob
                final_prediction_list.append(pred)
            i += 1

    final_pred = sum(final_prediction_list) / len(final_prediction_list)
    return root, final_pred


class SX_sGBDT:
    def __init__(
        self,
        n_estimator=1,
        depth=1,
        sx_order=(1, 0, 0),
        sx_seas_order=None,
        exog=None,
        n_data=None,
    ):

        self.n_data = n_data
        # sSGDT part
        if exog is None:
            warnings.warn("No exogenous variable passed; sGBDT will be inactive")
            self._has_sgbdt = False
            self.exog = None
        else:
            self.exog = np.asarray(exog)
            assert self.exog.ndim == 2, "non-2D exog"
            self._has_sgbdt = True
            self.n_estimators = n_estimator
            self.depth = depth
            self.n_node = 2 ** (self.depth + 1) - 1
            self.leaf_node = 2**self.depth
            self.innder_node = 2**self.depth - 1

        # SARIMAX
        self.sx_p, self.sx_d, self.sx_q = sx_order

        # Seasonal Part
        if sx_seas_order is not None:
            self.sx_P, self.sx_D, self.sx_Q, self.sx_m = sx_seas_order
        else:
            self.sx_P, self.sx_D, self.sx_Q, self.sx_m = 0, 0, 0, None

        state_dim = self.sx_p + self.sx_q + self.sx_P + self.sx_Q

        if self._has_sgbdt:
            state_dim += exog.shape[1]
            for i in range(self.n_estimators):
                state_dim += self.n_node * (exog.shape[1] * 1)  # weight
                state_dim += self.n_node * 1  # bias

        self.state_vector = rng.normal(size=(state_dim,))

        self.G_t = np.identity(state_dim, dtype=float)
        self.C_t = np.identity(state_dim, dtype=float)
        self.I = np.identity(state_dim, dtype=float)
        self.P_t = rng.normal(scale=UPD_NOISE_STD, size=(1, state_dim))
        self.Z_t_1 = np.identity(state_dim, dtype=float) / 10

        self._measurements = []
        self._errors = []
        self._predictions = []

    def _sx_state_transition(self):
        sx_size = self.sx_p + self.sx_q + self.sx_P + self.sx_Q
        if self.exog is not None:
            sx_size += self.exog.shape[1]
        self.state_vector[:sx_size] = rng.normal(scale=UPD_NOISE_STD, size=(sx_size))

    def _sgbdt_state_transition(self):
        sx_size = self.sx_p + self.sx_q + self.sx_P + self.sx_Q
        if self.exog is not None:
            sx_size += self.exog.shape[1]
        sgbdt_size = self.n_estimators * (self.n_node * (self.exog.shape[1] + 1))
        self.state_vector[sx_size : sx_size + sgbdt_size] = rng.normal(scale=UPD_NOISE_STD, size=(sgbdt_size))

    def _state_transition(self, t):
        """
        Perform the state transition equation, i.e.,
        \vec{s}_t = \vec{s}_{t-1} + \vec{e}_t
        """
        self._sx_state_transition()
        if self._has_sgbdt:
            self._sgbdt_state_transition()

    def get_state(self):
        return self.state_vector

    def put_state(self, state_vector):
        self.state_vector = state_vector

    def get_state_cov(self):
        return self.Z_t_1

    def put_state_cov(self, Z_t):
        self.Z_t_1 = Z_t

    def take_prediction(self, W_list, b_list, max_depth, input):

        weigths = list(zip(W_list, b_list))
        nums = list(range(1, 2 ** (max_depth + 1)))
        root, final_pred = buildTree(nums, weigths, input, max_depth)
        return final_pred

    def _sx_prediction(self, t):
        # Form r_t := [y_{t-1}, ..., y_{t-p},
        #              y_{t-m}, ..., y_{t-mP},
        #              e_{t-1}, ..., e_{t-q},
        #              e_{t-m}, ..., e_{t-mQ}]
        r_t = []

        # Check if we are at the start yet; fill with 0 if so. Otherwise,
        # last p/q items are taken.
        n_meas, n_errs = len(self._measurements), len(self._errors)

        # AR part
        if n_meas < self.sx_p:
            r_t.extend(self._measurements[::-1] + [0] * (self.sx_p - n_meas))
        elif self.sx_p:  # can say `else` as well but this saves a bit of time
            r_t.extend(self._measurements[: -1 - self.sx_p : -1])

        # Seasonal AR part
        _sx_has_seasonal_part = self.sx_P != 0 or self.sx_Q != 0
        if _sx_has_seasonal_part:
            # say m = 12, P = 3
            # so, need -12, -24, -36th values
            if n_meas < self.sx_m * self.sx_P:
                r_t.extend(
                    self._measurements[-self.sx_m :: -self.sx_m]
                    + [0] * np.ceil(self.sx_P - n_meas / self.sx_m).astype(int)
                )
                assert self.sx_P - n_meas / self.sx_m > 0, "mP versus n_meas gone wrong"
            elif self.sx_P:
                r_t.extend(self._measurements[-self.sx_m : -1 - self.sx_m * self.sx_P : -self.sx_m])

        # MA part
        if n_errs < self.sx_q:
            r_t.extend(self._errors[::-1] + [0] * (self.sx_q - n_errs))
        elif self.sx_q:
            r_t.extend(self._errors[-1 : -1 - self.sx_q : -1])

        # Seasonal MA part
        if _sx_has_seasonal_part:
            if n_errs < self.sx_m * self.sx_Q:
                r_t.extend(
                    self._errors[-self.sx_m :: -self.sx_m] + [0] * np.ceil(self.sx_Q - n_errs / self.sx_m).astype(int)
                )
                assert self.sx_Q - n_errs / self.sx_m > 0, "mQ versus n_errs gone wrong"
            elif self.sx_Q:
                r_t.extend(self._errors[-self.sx_m : -1 - self.sx_m * self.sx_Q : -self.sx_m])

        # Exogenous part
        if self.exog is not None:
            r_t.extend(self.exog[t])

        sx_size = self.sx_p + self.sx_q + self.sx_P + self.sx_Q
        if self.exog is not None:
            sx_size += self.exog.shape[1]
        sx_state_vector = self.state_vector[:sx_size]
        sx_pred = sx_state_vector @ np.array(r_t)
        return sx_pred

    def _sgbdt_prediction(self,x_t):

        n_estimators, depth, n_node, leaf_node, n_feature = (
            self.n_estimators,
            self.depth,
            self.n_node,
            self.leaf_node,
            self.exog.shape[1],
        )
        state_vector = self.state_vector

        W_list = [0] * n_node * n_estimators
        b_list = [0] * n_node * n_estimators

        sx_offset = self.sx_p + self.sx_q + self.sx_P + self.sx_Q + n_feature
        for i in range(n_estimators):
            estimators_offset = i * n_node
            for j in range(n_node):
                W_list[j + i * n_node] = state_vector[
                    sx_offset + estimators_offset + j * n_feature : sx_offset + estimators_offset + (j + 1) * n_feature,
                ]

        soft_tree_weight_ofset = n_estimators * n_node * n_feature
        for i in range(n_estimators):
            estimators_offset = i * n_node
            for j in range(n_node):
                b_list[j + i * n_node] = state_vector[
                    sx_offset
                    + soft_tree_weight_ofset
                    + estimators_offset
                    + j * 1 : sx_offset
                    + soft_tree_weight_ofset
                    + estimators_offset
                    + (j + 1) * 1,
                ]

        pred = self.take_prediction(W_list, b_list, depth, x_t)
        return pred

    def prediction(self, h_t, x_t, z_t, t):
        sx_pred_t = self._sx_prediction(t)
        sgbdt_pred_t = self._sgbdt_prediction(x_t) if self._has_sgbdt else 0
        y_pred_t = sx_pred_t + sgbdt_pred_t
        self._predictions.append(y_pred_t)

        return y_pred_t


def ekf(model, y_t, x_t, t, dt):
    model._measurements.append(y_t)
    h_t = None  # TODO

    model._state_transition(t)  # z_t Before EKF

    z_t = model.get_state()
    Z_t_1 = model.get_state_cov()
    Z_t = model.G_t @ Z_t_1 @ model.G_t.T + (model.C_t)

    forecast_y = model.prediction(h_t, x_t, z_t, t)
    r_t = y_t - forecast_y
    model._errors.append(r_t)
    print(f"Residual: {r_t} for observation {y_t}")
    r_t = np.array([[r_t]])

    R_t = model.P_t @ Z_t @ model.P_t.T
    K_t = Z_t @ model.P_t.T @ np.linalg.inv(R_t)

    z_t = z_t + (K_t @ r_t).squeeze()
    model.put_state(z_t)
    Z_t = (model.I - K_t @ model.P_t) @ Z_t
    model.put_state_cov(Z_t)


def EKF_SGBDT(y_obs, model):
    dt = 1
    X = model.exog

    for t, y_t in enumerate(y_obs):
        print(f"Time {t+1}")
        ekf(model=model, y_t=y_t, x_t=X[t], t=t, dt=dt)
    


def delhi_climate():
    df = pd.read_csv("DailyDelhiClimate.csv")
    df = df.drop(["date"], axis=1)
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    test_size = int(len(df) * 0.3)
    return X, y, test_size


