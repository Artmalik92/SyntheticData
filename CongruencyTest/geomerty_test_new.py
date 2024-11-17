import numpy as np

def geometric_chi_test_calc2(time_series_frag, sigma, sigma_0): #переделка массива с координатами вобщий вектор свлбодных членов
    L=np.zeros((time_series_frag.shape[0]*3))

    for i in range (time_series_frag.shape[0]):
        for j in range (3):
            L[3*i]=time_series_frag[0][i]
            L[3*i+1]=time_series_frag[1][i]
            L[3*i+2]=time_series_frag[2][i]

    # задаем массив эпох
    t=np.arange(0, time_series_frag.shape[0], 1)

  # цикл формирования матрицы коэффициентов
    for m in range (time_series_frag.shape[0]):
        ti=t[m]
        if m==0:
            A=np.hstack((np.identity(3)*ti, np.identity(3)))
        else:
            Aux=np.hstack((np.identity(3)*ti, np.identity(3)))
            A=np.vstack((A, Aux))

    print(np.linalg.cond(A))

    #формирование матрицы весов
    P=np.diag(sigma)/sigma_0 # сюда нужно вложить матрицу ковариационную
   
    #print(P.size, A.size)
    # решаем СЛАУ
    N = A.transpose().dot(np.linalg.inv(P)).dot(A)
    X = np.linalg.inv(N).dot(A.transpose().dot(np.linalg.inv(P)).dot(L)) # вектор параметров кинематической модели
    x_LS=np.array([X[0]*t[-1]+X[3], X[1]*t[-1]+X[4], X[2]*t[-1]+X[5]])
 
    #вычисляем вектор невязок
    V=A.dot(X)-L

    # СКП единицы веса
    mu= np.sqrt(np.sum(V.transpose().dot(np.linalg.inv(P)).dot(V))/(V.shape[0]-6))
    Qx=np.linalg.inv(N)*mu**2
    Qv=(Qx[0:3,0:3]*t[-1]*t[-1]+Qx[3:6,3:6])
    print(Qv)
    return (x_LS,Qv,mu, Qx)

