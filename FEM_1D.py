import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0

# np.set_printoptions(suppress=True)

class FEM_1D():
    '''
        Modelo de elementos finitos 1D para solução do problema de um capacitor
        preenchido por dois dielétricos de permissividades e1 e e2.
        Considerando a placa inferior do capacitor (1) está aterrada.

        É assumido que L >> d para que o problema seja reduzido a 1 dimensão. 
    '''
    def __init__(self, L, d1, d2, er1, er2, N1, N2, V0, p01 = 0, p02 = 0):
        ''' 
            Entradas:
                L   => Comprimento da placa do capacitor
                d1  => Altura do dielétrico 1
                d2  => Altura do dielétrico 2 
                er1 => Permissividade relativa do dielétrico 1
                er2 => Permissividade relativa do dielétrico 2
                N1  => Quantidade de segmentos no dielétrico 1
                N2  => Quantidade de segmentos no dielétrico 2
                V0  => Valor da diferença de potencial entre as placas do capacitor (Vsuperior - Vinferior)
                p01 => Densidade volumétrica de cargas livres no dielétrico 1 
                p02 => Densidade volumétrica de cargas livres no dielétrico 2
        '''
        self.L  = L
        self.d1 = d1
        self.d2 = d2
        self.e1 = er1 * epsilon_0
        self.e2 = er2 * epsilon_0
        self.N1 = N1
        self.N2 = N2
        self.V0 = V0
        self.p01 = p01
        self.p02 = p02

        self.N  = N1 + N2

        self.l1 = None
        self.l2 = None

        self.K  = None
        self.K1 = None
        self.K2 = None

        self.f  = None
        self.f1 = None
        self.f2 = None

        self.b  = None

        self.sol = None

    def solve(self):
        self.__calculateLengthElements()
        self.__calculateKi()
        self.__calculatefi()

        self.__calculateK()
        self.__calculatef()
        
        self.__applyBoundaryCondition()
        self.__calculateVs()
        
        return np.copy(self.sol)

    def __calculateLengthElements(self):
        '''
            Calcula o comprimento dos elementos do dielétrico 1 (l1)
            e do dielétrico 2 (l2).
        '''
        self.l1 = self.d1/self.N1
        self.l2 = self.d2/self.N2

    def __calculateKi(self):
        '''
            Calcula a matriz Ki para os elementos do dielétrico 1 (K1)
            e para os elementos do dielétrico 2 (K2).
        '''
        self.K1 = self.e1/self.l1 * np.array([[1, -1], [-1, 1]])
        self.K2 = self.e2/self.l2 * np.array([[1, -1], [-1, 1]])

    def __calculateK(self):
        '''
            Calcula a matriz K
        '''
        n = self.N + 1
        K = np.zeros((n,n))
        
        for i in range(self.N1):
            K[i][i]     += self.K1[0][0]
            K[i][i+1]   += self.K1[0][1]
            K[i+1][i]   += self.K1[1][0]
            K[i+1][i+1] += self.K1[1][1]
        
        for i in range(self.N1, self.N):
            K[i][i]     += self.K2[0][0]
            K[i][i+1]   += self.K2[0][1]
            K[i+1][i]   += self.K2[1][0]
            K[i+1][i+1] += self.K2[1][1]

        self.K = K
        print('-----------------------')
        print('K:')
        print(pd.DataFrame(self.K))    

    def __calculatefi(self):
        '''
            Calcula a matriz fi para os elementos do dielétrico 1 (f1)
            e para os elementos do dielétrico 2 (f2)
        '''
        self.f1 = -self.p01*self.l1/2 * np.array([1, 1])
        self.f2 = -self.p02*self.l2/2 * np.array([1, 1])
    
    def __calculatef(self):
        '''
            Calcula a matriz coluna f
        '''
        n = self.N + 1
        f = np.zeros(n)

        for i in range(self.N1):
            f[i]    += self.f1[0]
            f[i+1]  += self.f1[1]
        
        for i in range(self.N1, self.N):
            f[i]    += self.f2[0]
            f[i+1]  += self.f2[1]
        
        self.f = f
        print('-----------------------')
        print('f:')
        print(pd.DataFrame(self.f))    

    def __applyBoundaryCondition(self):
        '''
            Aplicando as condições de contorno para o problema:
            1) V1(0)        = 0
            2) V2(d1+d2)    = V0

            Ao aplicar essas condições de limites ao problema o vetor d se reduzirar a um vetor 
            com todos elementos nulos, logo não o consideraremos para o cálculo.
        '''
        b = np.copy(self.f)
        # Aplicando condição de contorno 1) V1(0) = 0
        c_v1 = self.K[:, 0]
        b += -c_v1 * 0

        # Aplicando condição de contorno 2) V2(0) = 0
        c_vN_plus_1 = self.K[:, self.N]
        b += -c_vN_plus_1 * self.V0

        # removendo a primeira e última linha e a primeira e última coluna da matriz K
        self.K = self.K[1:self.N, 1:self.N]
        b = b[1: self.N]
        self.b = b

        print('-----------------------')
        print('K após condição de contorno:')
        print(pd.DataFrame(self.K))
        print('-----------------------')
        print('b (f+d) após condição de contorno:')
        print(pd.DataFrame(self.b))


    def __calculateVs(self):
        '''
            Solução do sistema linear da forma KV = b
        '''
        Vs = np.linalg.solve(self.K, self.b)
        self.sol = Vs

    def calculateCapacitance(self):
        E = (self.sol[0] - self.sol[1])/self.l1
        sigma = -E*self.e1
        A = self.L*self.L
        Q = sigma * A
        C = Q/self.V0

        return C

def showPlot(d1, d2, N1, N2, V0, vs):
    x1 = np.linspace(0, d1, N1+1)
    x2 = np.linspace(d1, d1 + d2, N2+1)
    x = np.unique(np.concatenate((x1, x2)))
    
    y = np.copy(vs)
    y = np.concatenate((np.array([0]),y ,np.array([V0])))

    # print (y)
    # print (x)
    print(x)
    plt.plot(x, y, 'o')
    plt.show()

def showSol(sol, N, V0):
    vs = np.concatenate((np.array([0]),sol ,np.array([V0])))
    
    col_indx = []
    print('-------------------------------')
    print('Valores de potencial entre as placas:')
    for i in range(N+1):
        print(f'v{i+1} = {vs[i]}')
        


def main():
    L=0.02 
    d1=0.001
    d2=0.001
    er1=2
    er2=4
    N1=100
    N2=100
    V0=1
    model = FEM_1D(L, d1, d2, er1, er2, N1, N2, V0)
    sol = model.solve()

    showSol(sol, N1+N2, V0)
    # showPlot(d1, d2, N1, N2, V0, sol)
    C = model.calculateCapacitance()
    print(C)

if __name__ == '__main__':
    main()
