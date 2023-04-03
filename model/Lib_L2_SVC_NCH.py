
import numpy as np
from numpy import linalg
import math
from sklearn.svm import SVC, NuSVC

#------------------------------------------------------------------------
'''
    L2_SVC_NCH L2-SVC转化版本
'''
class L2_SVC_NCH_ByL2SVC(SVC):
    def __init__(self, 
                 kernel_mat, 
                 kernel_name='kernel_gaussian', 
                 gamma=5.0, 
                 C=np.PINF, 
                 myC=1.0, 
                 max_iter=2000, 
                 tol=1e-3, 
                 need_optValue=True):
        super().__init__(kernel=kernel_mat, 
                         gamma=gamma, 
                         C=np.PINF, 
                         max_iter=max_iter, 
                         tol=tol)
        self.myC = myC # C：用于求解L2-SVC；myC：用于L2-SVC-NCH计算p、q
        self.kernel_name = kernel_name
        kernel_dict = {'kernel_gaussian': self.kernel_gaussian, # 核函数字典
                       'kernel_linear': self.kernel_linear,
                       'kernel_quadratic': self.kernel_quadratic}
        self.mykernel = kernel_dict[kernel_name] # 核函数
        self.need_optValue = need_optValue

    def fit(self, X, y):
        self._K = self._gram_matrix(X)
        super().fit(X, y)
        M = np.sum(np.abs(self.dual_coef_.flatten())) / 2.0
        self.alpha_spv = np.abs(self.dual_coef_.flatten()) / M # L2-SVC的alpha 转化为 L2-SVC-NCH的alpha
        self.y_spv = y[self.support_]
        self.spv = X[self.support_]
        
        mask_positive = self.y_spv > 0
        mask_negative = ~ mask_positive
        alpha_spv_positive = self.alpha_spv[mask_positive] # 正类alpha
        alpha_spv_negative = self.alpha_spv[mask_negative] # 正类alpha
        ix_spv_positive = self.support_[mask_positive] # 正类全局索引
        ix_spv_negative = self.support_[mask_negative] # 负类全局索引

        # 计算p (正例)、q (负例)
        self.p = self.get_pq(y, alpha_spv_positive, ix_spv_positive)
        self.q = self.get_pq(y, alpha_spv_negative, ix_spv_negative)

        # 计算目标函数值
        if self.need_optValue:
            self.opt_value = self.get_optValue(self.support_, self.y_spv, self.alpha_spv)
        # self.optValue = 0.5 * np.dot(self.alpha_spv.T, self._K[self.support_][self.support_], self.alpha_spv)
        # print('正例alpha之和为：{0}，负例alpha之和为{1}'.format(np.sum(alpha_spv_positive), np.sum(alpha_spv_negative)))

    # 核矩阵计算
    def _gram_matrix(self, X):
        # 高斯核优化
        if self.kernel_name == 'kernel_gaussian':
            return np.exp(-self.gamma * ((X**2).sum(1).reshape(-1, 1) + (X**2).sum(1) - 2 * X @ X.T))
        else:
            m, _ = X.shape
            K = np.zeros((m, m), dtype=np.float64)
            for i in range(m):
                for j in range(i, m):
                    K[i][j] = self.mykernel(X[i], X[j])
                    K[j][i] = K[i][j]
            return K

    # 计算p（正类）,q（负类）
    def get_pq(self, y, alpha, ix_list):
        p = 0.0
        for i, ix in enumerate(ix_list):
            p += y[ix] * alpha[i] * (1.0 / self.myC)
            p += np.sum(alpha * y[ix_list] * self._K[ix, ix_list])
        # 取平均
        return p / len(ix_list)


    # 返回目标函数值
    def get_optValue(self, ix_list, y, alpha):
        opt_value = 0.0
        for i, ix_i in enumerate(ix_list):
            for j, ix_j in enumerate(ix_list):
                opt_value += alpha[i] * alpha[j] * y[i] * y[j] * self._K[ix_i][ix_j]
        opt_value += (1.0 / self.C) * np.sum(alpha * alpha)
        opt_value /= 2
        return opt_value
        
    # 预测函数
    def project(self, X):
        ''''''
        y_pred = np.zeros(len(X))
        for i in range(len(X)):
            s = 0.0
            for _spv, _y_spv, _alpha_spv in zip(self.spv, self.y_spv, self.alpha_spv):
                s += _y_spv * _alpha_spv * self.mykernel(X[i], _spv)
            y_pred[i] = s
        y_pred = y_pred - (self.p + self.q) / 2
        return y_pred

    # 预测
    def predict(self, X):
        ''''''
        # print('开始预测：')
        return np.sign(self.project(X))

    # 高斯核函数
    def kernel_gaussian(self, x1, x2):
        return np.exp(-self.gamma * (linalg.norm(x1 - x2) ** 2))
        # return np.exp(-linalg.norm(x1-x2) ** 2 / (2 * (self.gamma ** 2)))
    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)
    def kernel_quadratic(self, x1, x2):
        return (np.dot(x1, x2.T) ** 2)
    
# --------------------------------------------------------------------
'''
    L2_SVC_NCH 原生python版本
'''
class L2_SVC_NCH_Python():
    def __init__(self, 
                 kernel_name='kernel_gaussian', 
                 C=1.0, 
                 gamma=5.0, 
                 max_iter=2000, 
                 epsilon=1e-3, 
                 need_optValue=True):
        self.C = C # 惩罚系数
        self.gamma = gamma # 高斯参数
        self.max_iter = max_iter # 最大迭代次数
        self.epsilon = epsilon # 误差精度
        self.kernel_name = kernel_name
        kernel_dict = {'kernel_gaussian': self.kernel_gaussian, # 核函数字典
                       'kernel_linear': self.kernel_linear,
                       'kernel_quadratic': self.kernel_quadratic}
        self.kernel = kernel_dict[kernel_name] # 核函数
        self.need_optValue = need_optValue

    def fit(self, X, y):
        ''''''
        self.m, _ = X.shape # m样本数，n特征数
        self.K = self._gram_matrix(X) # 计算Kij
        self.G = np.outer(y.T, y) * self.K # Gij = yi*yj*Kij
        self.count = 0 # 计数器 记录迭代次数

        # 生成用于区分正类和负类的mask
        mask_positive = (y == 1) 
        mask_negative = ~ mask_positive 
        # 正类和负类的全局索引列表
        ix = np.arange(len(y))
        ix_positive = np.arange(len(y))[mask_positive]
        ix_negative = np.arange(len(y))[mask_negative]
        # alpha初始化
        # alpha = self.alpha_init(alpha, ix_positive, ix_negative)
        alpha = np.ones(len(y), dtype=np.float64)
        alpha[ix_positive] *= 1.0 / len(ix_positive)
        alpha[ix_negative] *= 1.0 / len(ix_negative)
        # kkt and picked
        iskkt, is_picked_all = False, False
        # print('开始求解alpha by SMO...')
        while True:
            # 计数
            self.count += 1
            # print('迭代次数：', count)
            # alpha_pre = np.copy(alpha) # 深拷贝（'='是地址引用）用于后面的比较判别停机条件
            
            if iskkt == False and is_picked_all == False:
                # Maximal Violoting Pair
                i, j, class_name, iskkt, is_picked_all = self.get_ij_byMVP(alpha, ix_positive, ix_negative)
                alpha_old_i, alpha_old_j = alpha[i], alpha[j]
                # print('iskkt：                     ', iskkt)
                # print('class_name：                ', class_name)
                # print('j，           i：           ', j, i)
                # print('alpha[j],     alpha[i]：    ', alpha[j], alpha[i])
            if iskkt:
                print('已满足kkt！')
                break
            if is_picked_all:
                print('已无违反对！')
                break
            if class_name == 'positive': # 处理正类
                alpha[j] = self.get_alpha_j(i, j, alpha, y) # 计算alpha_j unclip
                # clip
                if alpha[j] > alpha_old_i + alpha_old_j:
                    alpha[j] = alpha_old_i + alpha_old_j
                    alpha[i] = 0
                elif alpha[j] < 0:
                    alpha[j] = 0
                    alpha[i] = alpha_old_i + alpha_old_j
                else:
                    # alpha[j] = alpha[j]
                    alpha[i] = alpha_old_i + alpha_old_j - alpha[j]
                # alpha[j] = self.cilp(alpha[j]) # clip
                # if math.isclose(alpha[j], 1): # 如果alpha_j为1，则其他alpha置0
                #     alpha = self.clear_alpha_except_j(j, alpha, ix_positive)
                # alpha[i] = self.get_alpha_i(i, j, alpha, alpha[j], ix_positive) # 计算alpha_i 
            
            elif class_name == 'negative': # 处理负类
                alpha[j] = self.get_alpha_j(i, j, alpha, y) # unclip
                # clip
                if alpha[j] > alpha_old_i + alpha_old_j:
                    alpha[j] = alpha_old_i + alpha_old_j
                    alpha[i] = 0
                elif alpha[j] < 0:
                    alpha[j] = 0
                    alpha[i] = alpha_old_i + alpha_old_j
                else:
                    # alpha[j] = alpha[j]
                    alpha[i] = alpha_old_i + alpha_old_j - alpha[j]
                # alpha[j] = self.cilp(alpha[j])
                # if math.isclose(alpha[j], 1):
                #     alpha = self.clear_alpha_except_j(j, alpha, ix_negative)
                # alpha[i] = self.get_alpha_i(i, j, alpha, alpha[j], ix_negative)
            else:
                print('error!')
            # print('alpha[j]_new, alpha[i]_new：', alpha[j], alpha[i])
            # print('🔺alpha[j]， 🔺alpha[i]：  ', alpha[j] - alpha_pre[j], alpha[i] - alpha_pre[i])     
            # print('alpha[ix_positive]：', alpha[ix_positive])
            # print('alpha[ix_negative]：', alpha[ix_negative])
           
            # 计算difference
            # diff = np.linalg.norm(alpha - alpha_pre)
            # # print('alpha diff：', diff)
            # if diff < 1e-7:
            #     break

            # 最大迭代次数
            if self.count >= self.max_iter:
                print('已超过最大迭代次数{0}！'.format(self.max_iter)) 
                break
            
        # print('alpha求解结束')

        # 支持向量布尔列表
        self.mask_spv = alpha > 1e-6 
        # 获取支持向量索引、支持向量、y和alpha
        self.ix_spv = np.arange(len(y))[self.mask_spv]
        self.spv = X[self.mask_spv]
        self.y_spv = y[self.mask_spv]
        self.alpha_spv = alpha[self.mask_spv]

        # 获取正例和负例支持向量索引
        ix_spv_positive = np.arange(len(y))[self.mask_spv & mask_positive]
        ix_spv_negative = np.arange(len(y))[self.mask_spv & mask_negative]

        # 计算p (正例)、q (负例)
        self.p = self.get_pq(y, alpha, ix_spv_positive)
        self.q = self.get_pq(y, alpha, ix_spv_negative)

        if self.need_optValue:
            self.opt_value = self.get_optValue(self.ix_spv, self.y_spv, self.alpha_spv)
        # print('opt_value: ', self.opt_value)
        # print('正例alpha之和为：{0}，负例alpha之和为{1}'.format(np.sum(alpha[ix_spv_positive]), np.sum(alpha[ix_spv_negative])))

    # 计算p（正类）,q（负类）
    def get_pq(self, y, alpha, ix):
        p = 0.0
        for i in ix:
            p += y[i] * alpha[i] / self.C
            p += np.sum(alpha[ix] * y[ix] * self.K[i, ix])
        # 取平均
        return p / len(ix)

    # Maximal Violoting Pair（最大违反对原则）选取alpha_i, alpha_j
    def get_ij_byMVP(self, alpha, ix_positive, ix_negative):
        # 初始化
        i, j = 0, 0
        ix_m_positive, ix_M_positive, ix_m_negative, ix_M_negative = 0, 0, 0, 0
        m_positive, M_positive, m_negative, M_negative = 0.0, 0.0, 0.0, 0.0
        class_name = '' # 本次违反对选取的类比：正类or负类
        is_kkt = False
        is_pick_all_positive, is_pick_all_negative = True, True # 防止出现空候选集

        # 待选取列表
        _list = - np.dot((self.G + (1.0 / self.C) * np.identity(self.m)), alpha)
        
        # 定义候选集：索引集合
        ix_positive_up = ix_positive[alpha[ix_positive] < 1]
        ix_positive_down = ix_positive[alpha[ix_positive] > 1e-6]
        ix_negative_up = ix_negative[alpha[ix_negative] > 1e-6]
        ix_negative_down = ix_negative[alpha[ix_negative] < 1]
        
        # 求最大违反对及对应的索引
        if ix_positive_up.size != 0 and ix_positive_down.size != 0:
            _ix_m_positive = np.argmax(_list[ix_positive_up])
            ix_m_positive = ix_positive_up[_ix_m_positive]
            m_positive = _list[ix_m_positive]
            _ix_M_positive = np.argmin(_list[ix_positive_down])
            ix_M_positive = ix_positive_down[_ix_M_positive]
            M_positive = _list[ix_M_positive]
            is_pick_all_positive = False
        
        if ix_negative_up.size != 0 and ix_negative_down.size != 0:
            _ix_m_negative = np.argmax(_list[ix_negative_down])
            ix_m_negative = ix_negative_down[_ix_m_negative]
            m_negative = _list[ix_m_negative]
            _ix_M_negative = np.argmin(_list[ix_negative_up])
            ix_M_negative = ix_negative_up[_ix_M_negative]
            M_negative = _list[ix_M_negative]
            is_pick_all_negative = False

        # if self.count == self.max_iter-1:
        #     # print('iter: {0}, opt_value: {1}'.format(self.count, self.opt_value))
        #     # print('iter, m_+, M_+, m_-, M_-', self.count, m_positive, M_positive, m_negative, M_negative)
        #     print('iter: {0}, (m_+ - M_+): {1}, (m_- - M_-): {2}'.format(self.count, (m_positive - M_positive),(m_negative - M_negative)))
        
        # 满足KKT条件
        if m_positive <= M_positive + self.epsilon and m_negative <= M_negative + self.epsilon:
            is_kkt = True
        # 对比正、负违反对，选取违法程度更大的那类作为返回
        else:
            if is_pick_all_positive and is_pick_all_negative:
                class_name = 'none'
            elif is_pick_all_positive or is_pick_all_negative:
                class_name = 'positive' if is_pick_all_positive == False else 'negative'
            else:
                class_name = 'positive' if (m_positive - M_positive) > (m_negative - M_negative) else 'negative'
            i, j = (ix_m_positive, ix_M_positive) if class_name == 'positive' else (ix_m_negative, ix_M_negative)

        return i, j, class_name, is_kkt, (is_pick_all_positive and is_pick_all_negative)

    # clear_alpha_except_j
    # def clear_alpha_except_j(self, j, alpha, ix_list):
    #     w = np.ones(len(alpha), dtype=np.float64)
    #     w[ix_list] = 0.0
    #     new_alpha = w * alpha
    #     new_alpha[j] = 1.0
    #     return new_alpha

    # 计算alpha_j
    def get_alpha_j(self, i, j, alpha, y):
        ''''''        
        # w相当于掩码作用，将i和j处置0（不参与计算）
        w = np.ones(len(y))
        w[i], w[j] = 0, 0
        formula_up = (alpha[i] + alpha[j]) * self.K[i][i] \
            - (alpha[i] + alpha[j]) * y[i] * y[j] * self.K[i][j] \
            + y[i] * np.sum(w * alpha * y * self.K[i, :]) \
            - y[j] * np.sum(w * alpha * y * self.K[j, :]) \
            + (1.0 / self.C) * (alpha[i] + alpha[j])
        formula_down = self.K[i][i] + self.K[j][j] - 2 * y[i] * y[j] * self.K[i][j] \
            + 2 * (1.0 / self.C)
        alpha_j = formula_up / formula_down
        return alpha_j

    # 裁剪
    # def cilp(self, alpha_j):
    #     ''''''
    #     if alpha_j < 1e-6:
    #         alpha_j = 0
    #     elif alpha_j > 1:
    #         alpha_j = 1
    #     else:
    #         pass
    #     return alpha_j

    # 获取alpha_i    
    # def get_alpha_i(self, i, j, alpha, alpha_j, ix_list):
    #     # 去除i, j
    #     w = np.ones(len(alpha))
    #     w[i], w[j] = 0, 0
    #     w = w[ix_list]
    #     alpha_i = 1 - np.sum(w * alpha[ix_list]) - alpha_j
    #     return alpha_i
    
    # 核矩阵计算
    def _gram_matrix(self, X):
        if self.kernel_name == 'kernel_gaussian':
            return np.exp(-self.gamma * ((X**2).sum(1).reshape(-1, 1) + (X**2).sum(1) - 2 * X @ X.T))
        else:
            m, _ = X.shape
            K = np.zeros((m, m))
            for i in range(m):
                for j in range(i, m):
                    K[i][j] = self.kernel(X[i], X[j])
                    K[j][i] = K[i][j]
            return K

    # 返回目标函数值
    def get_optValue(self, ix_list, y, alpha):
        opt_value = 0.0
        for i, ix_i in enumerate(ix_list):
            for j, ix_j in enumerate(ix_list):
                opt_value += alpha[i] * alpha[j] * y[i] * y[j] * self.K[ix_i][ix_j]
        opt_value += (1.0 / self.C) * np.sum(alpha * alpha)
        opt_value /= 2
        return opt_value

    # 预测函数
    def project(self, X):
        ''''''
        y_pred = np.zeros(len(X))
        for i in range(len(X)):
            s = 0.0
            for _spv, _y_spv, _alpha_spv in zip(self.spv, self.y_spv, self.alpha_spv):
                s += _y_spv * _alpha_spv * self.kernel(X[i], _spv)
            y_pred[i] = s
        y_pred = y_pred - (self.p + self.q) / 2
        return y_pred

    # 预测
    def predict(self, X):
        ''''''
        # print('开始预测：')
        return np.sign(self.project(X))

    # 高斯核函数
    def kernel_gaussian(self, x1, x2):
        return np.exp(-self.gamma * linalg.norm(x1 - x2) ** 2)
        # return np.exp(-linalg.norm(x1-x2) ** 2 / (2 * (self.gamma ** 2)))
    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)
    def kernel_quadratic(self, x1, x2):
        return (np.dot(x1, x2.T) ** 2)

# --------------------------------------------------------------------
'''
    L2_SVC_NCH Nu-SVC转化版
        - Lagrange系数存在偏差，暂不启用
'''
# class L2_SVC_NCH_ByNuSVC(NuSVC):
#     def __init__(self, kernel_mat, kernel_name='kernel_gaussian', gamma=5.0, C=1.0, nu=0.5, max_iter=2000, tol=1e-3):
#         super().__init__(kernel=kernel_mat, nu=nu, gamma=gamma, max_iter=max_iter, tol=tol)
#         self.C = C # 惩罚系数
#         kernel_dict = {'kernel_gaussian': self.kernel_gaussian, # 核函数字典
#                        'kernel_linear': self.kernel_linear,
#                        'kernel_quadratic': self.kernel_quadratic}
#         self.mykernel = kernel_dict[kernel_name] # 核函数

#     def fit(self, X, y):
#         ''''''
#         self._K = self._gram_matrix(X)
#         super().fit(X, y)

#         self.alpha_spv = np.abs(self.dual_coef_.flatten())
#         self.y_spv = y[self.support_]
#         self.spv = X[self.support_]
        
#         mask_positive = self.y_spv > 0
#         mask_negative = ~ mask_positive
#         alpha_spv_positive = self.alpha_spv[mask_positive] # 正类alpha
#         alpha_spv_negative = self.alpha_spv[mask_negative] # 正类alpha
#         ix_spv_positive = self.support_[mask_positive] # 正类全局索引
#         ix_spv_negative = self.support_[mask_negative] # 负类全局索引

#         # 计算p (正例)、q (负例)
#         self.p = self.get_pq(y, alpha_spv_positive, ix_spv_positive)
#         self.q = self.get_pq(y, alpha_spv_negative, ix_spv_negative)
        
#         print('正例alpha之和为：{0}，负例alpha之和为{1}'.format(np.sum(alpha_spv_positive), np.sum(alpha_spv_negative)))


#     # 核矩阵计算
#     def _gram_matrix(self, X):
#         m, _ = X.shape
#         K = np.zeros((m, m), dtype=np.float64)
#         for i in range(m):
#             for j in range(i, m):
#                 K[i][j] = self.mykernel(X[i], X[j])
#                 K[j][i] = K[i][j]
#         return K

#     # 计算p（正类）,q（负类）
#     def get_pq(self, y, alpha, ix_list):
#         p = 0.0
#         for i, ix in enumerate(ix_list):
#             p += y[ix] * alpha[i] * (1.0 / self.C)
#             p += np.sum(alpha * y[ix_list] * self._K[ix, ix_list])
#         # 取平均
#         return p / len(ix_list)

#     # 预测函数
#     def project(self, X):
#         ''''''
#         y_pred = np.zeros(len(X))
#         for i in range(len(X)):
#             s = 0.0
#             for _spv, _y_spv, _alpha_spv in zip(self.spv, self.y_spv, self.alpha_spv):
#                 s += _y_spv * _alpha_spv * self.mykernel(X[i], _spv)
#             y_pred[i] = s
#         y_pred = y_pred - (self.p + self.q) / 2
#         return y_pred

#     # 预测
#     def predict(self, X):
#         ''''''
#         # print('开始预测：')
#         return np.sign(self.project(X))

#     # 高斯核函数
#     def kernel_gaussian(self, x1, x2):
#         return np.exp(-self.gamma * (linalg.norm(x1 - x2) ** 2))
#         # return np.exp(-linalg.norm(x1-x2) ** 2 / (2 * (self.gamma ** 2)))

#     def kernel_linear(self, x1, x2):
#         return np.dot(x1, x2.T)
#     def kernel_quadratic(self, x1, x2):
#         return (np.dot(x1, x2.T) ** 2)
    
