### L2-SVC-NCH 工具包设计

---

#### Content

- [L2-SVC转化版](# L2-SVC-NCH L2-SVC转化版 - Done)
- [原生python版](# 原生python版 - Done)
- [实验对比](# 实验对比)



#### L2-SVC转化版 - Done

----

**设计概要**

- 本程序继承Sklearn的C-SVC模型（底层是LibSVM）可参考[learn.svm.SVC — scikit-learn 1.1.1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)、[LibSVM.pdf](extension://oikmahiipjniocckomdccmplodldodja/pdf-viewer/web/viewer.html?file=https%3A%2F%2Fwww.csie.ntu.edu.tw%2F~cjlin%2Fpapers%2Flibsvm.pdf)
- 修改了C-SVC的初始化，使之符合L2-SVC模型；
- 将L2-SVC求解得到的Lagrange系数$\alpha$转化为L2-SVC-NCH的Lagrange系数$\bar{\alpha}$。转化方法可参考Paper：[A Soft-Margin Support Vector Machine based on Normal Convex Hulls | Request PDF (researchgate.net)](https://www.researchgate.net/publication/272853868_A_Soft-Margin_Support_Vector_Machine_based_on_Normal_Convex_Hulls)；
- 后续的代码设计同论文一致。

**使用方法**

```python
# 包目录：model
# 文件名：Lib_L2_SVC_NCH.py
# class：L2_SVC_NCH_ByL2SVC
from model.Lib_L2_SVC_NCH import L2_SVC_NCH_ByL2SVC

# global parameter
C = 1.0
gamma = 5.0

# 自定义核矩阵 K' <- K + 1/C*I
def kernel_mat(X1, X2):
    K = np.zeros((len(X1), len(X2)), dtype=np.float64)
    for i in range(len(X1)):
        for j in range(i, len(X2)):
            K[i][j] = linalg.norm(X1[i] - X2[j]) ** 2
            K[j][i] = K[i][j]
    K *= -gamma
    K = np.exp(K)
    K += 1.0 / C * np.identity(len(X1)) 
    return K

# 初始化
model = Luo_SVM_BySklearnSVC(kernel_mat=kernel_mat, kernel_name='kernel_gaussian', gamma=gamma, C=np.PINF, myC=C, max_iter=2000, tol=1e-3, need_optValue=True)
# 训练
model.fit(X_train, y_train)
# 预测
y_pred = model.predict(X_test)
print('Accuracy of Luo_SVM by L2-SVM SKlearn: ', accuracy_score(y_test, y_pred))
# 获取目标函数值（loss）
model.opt_value
```

**其他**

- L2-SVC仅用于求解Lagrange系数



#### 原生python版 - Done

----

**L2-SVC-NCH**
$$
\begin{align*}
& min \ \frac{1}{2} \alpha^{T}(G+\delta I)\alpha \\
& st. \ \ \ \sum_{y_{i}=1}^{}\alpha _{i}=\sum_{y_{i}=-1}^{}\alpha _{i}=1\\ 
&\ \ \ \ \ \ \ \  0\le \alpha _{i} \le1, \ \ \ { {\small i=1,2,\dots,l} } 
\end{align*}
$$
**设计概要**

- 参考文章：[A Soft-Margin Support Vector Machine based on Normal Convex Hulls](https://www.researchgate.net/publication/272853868_A_Soft-Margin_Support_Vector_Machine_based_on_Normal_Convex_Hulls)

- 算法推导：使用SMO求解，可参考：[Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods ](http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.41.1639)

- 训练

  - $\alpha$初始化：均值化，即$\alpha_{i} = 1/l^{+}(y_{i} =1)$，$\alpha_{i} = 1/l^{-}(y_{i} =-1)$，$l^{+}和l^{-}分别表示正和负样本数量$

  - loop

    1）用最大违反对策略选取$\alpha_{i}和\alpha_{j}$（同为正类或同为负类）

    2）计算$\alpha_{j}^{new,unc}$ （计算公式可参考算法推导）

    3）$\alpha_{j}^{new} = clip(\alpha_{j}^{new,unc})$ （映射至(0,1)：超出1的部分置为1，低于0的部分置为0）

    4）计算$\alpha_{i}^{new} = 1-\alpha_{j}^{new}$

    5）判断停机条件（一定精度内满足KKT条件）

  - 计算$p、q$（具体参考论文）

- 预测
  $$
  D(x)=sign(\sum_{i=1}^{l}y_{i}\alpha_{i}^{*}k(x,x_{i})-(p^{*}+q^{*})/2 )
  $$

**使用方法**

```python
# 包目录：model
# 文件名：Lib_L2_SVC_NCH.py
# class：L2_SVC_NCH_ByPython_v1, L2_SVC_NCH_ByPython_v2
from model.Lib_L2_SVC_NCH import L2_SVC_NCH_ByPython_v1, L2_SVC_NCH_ByPython_v2

# 初始化
model = L2_SVC_NCH_ByPython_v1(kernel_name='kernel_gaussian', C=0.1, gamma=5.0, max_iter=2000, epsilon=1e-3, need_optValue=True)
# model = L2_SVC_NCH_ByPython_v2(kernel_name='kernel_gaussian', C=0.1, gamma=5.0, max_iter=2000, epsilon=1e-3, need_optValue=True) 
# 训练
model.fit(X_train, y_train)
# 预测
y_pred = model.predict(X_test)
print('Accuracy of Luo_SVM by SMO: ', accuracy_score(y_test, y_pred))
# 获取目标函数值（loss）
model.opt_value 
```

**版本说明**

- L2_SVC_NCH_ByPython_v1：标准版，按照算法步骤实现；
- L2_SVC_NCH_ByPython_v2：在v1版本的基础上，当最优解无法迭代时采取次优解选取策略，一定程度优化$\alpha$的迭代过程。



#### 实验对比

----

- Dataset

  |          NO          |   1   |     2      |     3      |    4    |   5    |
  | :------------------: | :---: | :--------: | :--------: | :-----: | :----: |
  |       Dataset        | Heart | Ionosphere | Australian | Diabets | German |
  | Number of instances  |  270  |    351     |    690     |   768   |  1000  |
  | Number of attributes |  13   |     33     |     14     |    8    |   24   |

- Result

  | Dataset\Method | L2-SVC-NCH(py_v1) | L2-SVC-NCH (py_v2) | L2-SVC-NCH(L2-SVC转化版) |    标准SVC    |
  | :------------: | :---------------: | :----------------: | :----------------------: | :-----------: |
  |     Heart      |   0.8148±0.0524   |   0.8407±0.0416    |      0.8296±0.0319       | 0.8222±0.0222 |
  |   Ionosphere   |   0.9211±0.0276   |   0.9267±0.0382    |      0.9296±0.0295       | 0.9296±0.0295 |
  |   Australian   |   0.829±0.0232    |   0.8217±0.0218    |       0.842±0.0148       | 0.8464±0.0133 |
  |    Diabets     |   0.7377±0.0177   |    0.726±0.027     |      0.7506±0.0301       | 0.7506±0.0274 |
  |     German     |   0.714±0.0111    |    0.724±0.0124    |       0.738±0.0196       | 0.746±0.0177  |
