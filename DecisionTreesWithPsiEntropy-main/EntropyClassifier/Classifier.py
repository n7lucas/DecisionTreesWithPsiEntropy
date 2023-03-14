# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:10:28 2023

@author: PC
"""

from BinaryTree.BinaryStructure import BinaryTree
import numpy as np
import pandas as pd
import math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import hamming_loss, accuracy_score 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler



class EntropyClassifier ():
    def __init__(self, optimizer='monte carlo', nb_iter =10000, nb_leaf = 0.01, nb_branch_max=20, activation='heaviside' ):
        self.activation = activation
        self.optimizer = optimizer
        self.nb_iter = nb_iter
        self.nb_leaf = nb_leaf
        self.nb_branch_max = nb_branch_max
        self.tree = None
        self.total_branches = 0
 
    def _random_a(self,X):
        """
        Just declared function of generate vector a, in each Straight Line version (P1, P2,...) i will override this function
        with the suitable version
        
        Receive: X (data train input)
        
        Return: Vector a based on lenght of X, P1 (a in R^n+1), P2 (a in R^3n),...
        
        """
        
        pass
    
    def _normalization(self, a):
        return(a/np.linalg.norm(a[0,1:],2))


    def _norm_vector (self,a):
        new_a = a.copy()
        normalized_v =  new_a / np.sqrt(np.sum( new_a[0,1:]**2))
        return  normalized_v
  
    """I gona call this function inside de class of classifier"""
    "Call all function that e«generate the line"
    def _Line(self,X,a):
        """
        No code on here, i will override in each straightLine class, with the suitable version
        
        Input: X -> input data train, y -> input data test, a -> vector a
        
        Output: D_classe -> dict of set of classes, f_pos -> positive frequency, f_neg -> negative frequency,
                            D_pos -> set of events in positive demiplan, D_neg -> set of events in negative demiplan,
                            S_a_pos -> Entropy of positive demiplan, S_a_neg -> Entropy of negative demiplan
                            S_a_Total -> Total Entropy
        """
        pass
    

    
    def _frequency_heaviside(self,D_neg,D_pos):
        
        """
        frequency with the heaviside function, divide the data in each intersection and the total cardinal of demiplan
        
        To avoid the divizion by zero problem we substitute 0 with a very small number(10^-12).
        
        Input: D_inter_pos -> dict of all classes intersection and the positive Demiplan
               D_inter_neg -> dict of all classes intersection and the negative Demiplan
               D_neg -> set of negative values
               D_pos -> set of positive values
        
        Output: f_pos -> total frequency of events in positive demiplan
                f_neg -> total frequency of events in negative demiplan
                 
                Note: the sum for each array frequency must be equal to 1.
        """
        f_pos = np.divide(np.unique(D_pos[:,-1], return_counts=True)[1], len(D_pos))
        f_neg = np.divide(np.unique(D_neg[:,-1], return_counts=True)[1], len(D_neg))
        return f_pos, f_neg
    
    
    def _parcial_entropy(self,f_pos, f_neg):
     
        """
        Calculation of parcial entropy of S+ and S-
        
        Input: f_pos -> dict of positive frequencies of all classes,
               f_neg -> dict of negative frequencies of all classes
               
        Output: S_a_pos -> Scalar value of Positive Entropy
                S_a_neg -> Scalar value of Negative Entropy
        """
   
        S_a_pos =  -np.dot(f_pos,np.log(f_pos))
        S_a_neg =  -np.dot(f_neg,np.log(f_neg))

        return S_a_pos, S_a_neg

    
    def _total_entropy_heavi(self,X,a):
        """
        Input: f_pos -> positive frequencies of each class in a dict
               f_neg -> negative frequencies of each class in a dict
               D_pos -> Set of all events presents in negative demiplan
               D_neg -> Set of all events presents in positive demiplan
               
        Output: S_a_pos -> Scalar positive Entropy
                S_a_neg -> Scalar negative Entropy
                S_a_Total -> Scalar of total Entropy
        """
        pass

    "Recupera a classe com mais aparições em cada folha"
    def __dominant_class(self,D_pos, D_neg): #Alterado
        """
        Input: D_pos -> Numpy array for data present in positive side of hyperplane
               D_neg -> Numpy array for data present in negative  side oh hyperplace
               
        Output: dominant_pos -> Class with highest appearance in D_pos (positive side of hyperplane)
                dominant_neg -> Class with highest appearance in D_neg (negative side of hyperplane)    
        
        If the number of highest apperance of a determinant class is equal to another (ex: 6 rows where 3 have class = 1 and 3 = 2)
        then we flip and take the first class of the dataset to be the highest one
        
        """
        try:
            dominant_pos = np.bincount(D_pos.astype(int)[:, -1]).argmax()
        except ValueError:
            dominant_pos = 10000#Em caso de nao 
        #else: dominant_pos = 5
        #if len(D_pos) !=0:
        try:
             dominant_neg = np.bincount(D_neg.astype(int)[:, -1]).argmax()
        except ValueError:
             dominant_neg = 10000#D_pos[0,-1]
        #else: dominant_neg = 5
        return dominant_pos, dominant_neg
    
    def __plot_accuracy_brench (self,best_params):
        brench_list = []
        accuracy_list = []
        for i in range (len(best_params)):
            brench_list.append(best_params[i][4])
            accuracy_list.append(best_params[i][8][1])
            #print("ENTROPIAASDSAFD", (best_params[0][8][0]))
        brench_list, accuracy_list = zip(*sorted(zip(brench_list, accuracy_list)))
        fig = plt.figure()
        plt.plot(brench_list, accuracy_list, '-o')
        plt.xlabel("Brench")
        plt.ylabel("Accuracy")
        plt.title("Accuracy x Brench")
        plt.show() 
    
    
    
    def __grafico_entropia_low (self,list_S, list_r):
           fig = plt.figure()
           plt.plot(list_r, list_S, '-o')
           plt.xlabel("r")
           plt.ylabel("S_min")
           plt.title("Entropia Loss")
           plt.show()
    
    def __grafico_vetora(self,plot_a, plot_b, plot_c, list_r):
           fig = plt.figure()
           plt.plot (list_r, plot_a, label="a_0")
           plt.plot(list_r, plot_b, label="a_1")
           plt.plot(list_r, plot_c, label="a_2")
           plt.legend()
           plt.xlabel("iteration")
           plt.ylabel("vector a0 a1 a2")
           plt.title("Vector Changes")
           plt.show
 
    
    def __monte_carlo(self,X): #Aqui x_train possui a classe concatenada
    
        """
        Input: X -> numpy arrray of all atributes
           y -> numpy array with all classes
           
        Output: best_params_calc -> Dict with values of frequency and entropies of the last node dataset split
                list_S -> List of Minimum entropy found en each node split (for decrease plot of entropy over iteration)
                list_r -> Capture of the iteration that have found the minimum entropy (for decrease plot of entropy over iter)
                best_a -> Best vector found that give us the lowest entropy for a given dataset
                f_pos, f_neg -> Scalar value that Calculated frequency result of the current vecotr a found
                D_pos, D_neg -> Set's of each demiplan thats contain the values that belong
                S_a_pos, S_a_neg -> Scalar value, Entropy of each side of the demiplan for the current vector a of separation
    
        """
        list_S = []
        list_r =[]
        S_min = 10**20
        best_a = self._random_a(X)
        for r in range (1,self.nb_iter):
            a = self._random_a(X)
            _ , _ , _ , _ , S_a_pos, S_a_neg,S_a_total = self._Line(X,a)
            if (S_a_total < S_min):
                S_min = S_a_total #Agora tambem vou pegar as entropias parciais para utilizalas na arvore
                best_a = a
                r_min = r #iteracao que obteve o melhor resultado
                list_S.append(S_min)
                list_r.append(r_min)
        f_pos, f_neg, D_pos, D_neg, S_a_pos, S_a_neg,S_a_total = self._Line(X,best_a)
        best_params_calc = {'f_neg' : f_neg, 'f_pos' : f_pos, 'S_a_pos' : S_a_pos, 'S_a_neg' : S_a_neg, 'S_total' : S_a_total }
        return  best_params_calc, list_S,list_r,best_a,f_pos, f_neg, D_pos, D_neg, S_a_pos, S_a_neg  
    
    
    
    """Return Leaf with the highest entropy value if satisfy de function stop_leaf"""   
    def __readable_leaf(self,Tree,nb_branch): # IOld   def __stop_leaf(self,leaf, grid_entropy, grid_leaf):
         """
         Input : root -> Structure of Binary Tree, nb_branch -> Number of branches in the binary tree
         
         
         Output: S_min -> (Scalar) Go through each leaf and find the node with the lowest entropy and get the entropy value
                 new_leaf -> (Node)return the node with lowest entropy
                 nb_current -> (Scalar) Current value of number of branches
            """
         S_min = math.inf
         new_leaf = 0
         leaf_min = 0
         for leaf in  Tree.Le():
             print(Tree.nodes[leaf])
             if (self.__stop_leaf(Tree.nodes[leaf]) == False):
                 if leaf_min < Tree.nodes[leaf].data["entropy"]:
                    leaf_min = Tree.nodes[leaf].data["entropy"]
                    new_leaf = Tree.nodes[leaf]
                    S_min = leaf_min
         nb_branch = nb_branch + 1
         return S_min,new_leaf,nb_branch
     
    "Verifica se a folha esta em condições de ser separada"
    def __stop_leaf(self, leaf):
            """
            Input: leaf -> (Node) Current leaf
            
            Output: (bool) True or false if the node can be separated
            """
            if np.unique(leaf.data["data"][:,-1]).size == 1: #If i only have one class in the node i dont have more reason to split, so i return TRUE
                return True
            if(leaf.data["data"].shape[0]  <= leaf.data["data"].shape[0]  * self.nb_leaf): 
                #print("aaaa", (leaf.name["Data"].shape[0]  * self.nb_leaf))#Passo o tamanho do banco de acordo com uma porcentagem dos dados
                return True
            else: 
               # print("ffffsdf", (leaf.name["Data"].shape[0]* self.nb_leaf))
                return False
    def run(self,X, y, x_test, y_test):
             self.nb_leaf = 0.01
             self.nb_branch_max = 20
             best_params_calc,list_s, list_r,self.tree, self.total_branches  = self._Tree(X,y) 
             return  best_params_calc,list_s, list_r  
   
    "Arvore binaria do dataset"            
    def _Tree(self, X, y):
        
        """
        Input: X -> (np array) Input Train attributes, y -> (np array) Input train class
        
        
         Output: best_params_calc -> Dict with values of frequency and entropies of the last node dataset split
                list_S -> List of Minimum entropy found en each node split (for decrease plot of entropy over iteration)
                list_r -> Capture of the iteration that have found the minimum entropy (for decrease plot of entropy over iter)
                rooot -> (node) Binary Tree structure
                nb_branch -> (Scalar) Number of leafs of the binary Tree Structure
        """
        best_params_calc_list = []
        list_s_list = []
        list_r_list =[]
        nb_branch = 0
        nb_branch_max = self.nb_branch_max
        df = np.concatenate((X,y),axis=1) #Estou juntando os atributos com a classe de predicao
        Tree = BinaryTree(1)
        root = Tree.nodes[1]
        keys =  ["data", "entropy"]
        values = [df,100]
        root.data = {k: v for k, v in zip (keys, values)}
        index_left, index_right = 2, 3
        S_min,new_leaf,nb_branch = self.__readable_leaf(Tree,nb_branch)
        while  ((S_min !=math.inf and nb_branch < nb_branch_max)):  #(S_min !=math.inf and root.height < 10)
            if (self.optimizer == "montecarlo"): #Depois resolver de maneira mais elegante    
                best_params_calc,list_s,list_r,best_a,f_pos, f_neg,D_pos,D_neg,S_pos_min, S_neg_min = self.__monte_carlo(new_leaf.data["data"]) 
                #print("TRI")
            pos_classe, neg_classe = self.__dominant_class(D_pos, D_neg)
            new_leaf.a = best_a
            Tree.add_Node(new_leaf.index, index_left,index_right )
            keys =  ["data", "entropy", "classe"]
            values = [D_neg,S_neg_min, neg_classe]
            Tree.nodes[index_left].data = {k: v for k, v in zip (keys, values)}
            
            keys2 =  ["data", "entropy", "classe"]
            values2 = [D_pos,S_pos_min, pos_classe]
            Tree.nodes[index_right].data = {k: v for k, v in zip (keys2, values2)}
            index_right = index_right +2
            index_left = index_left +2
            S_min,new_leaf,nb_branch = self.__readable_leaf(Tree,nb_branch)
        return best_params_calc_list,list_s_list, list_r_list,Tree, nb_branch

    
    "Calculo da funcao de psi dinamico para n atriburtos"    
    
    def _calcule_psi(self,row,a):
        pass

    def __find_label(self,x,threshold):
        node = self.tree.nodes[1]
        node.C_k.append(x)
        while self.tree.is_Leaf(node.index) == False: 
                node = self.__next_node(node,x,threshold)
                node.C_k.append(x)
        return node.data["classe"]
    
    def path(self,x,threshold):
        x_path = []
        node = self.tree.nodes[1]
        x_path.append(node)
        node.C_k.append(x)
        while self.tree.is_Leaf(node.index) == False: 
                node = self.__next_node(node,x,threshold)
                x_path.append(node)
                node.C_k.append(x)
        return x_path
    
    def leaf(self,x,threshold):
        listPath = self.path(x,threshold)
        return listPath[-1]
      

    "Percorre a arvore com os dados de testes ate a folha"
    def __next_node(self,node,row,threshold): #WI will create a variable Condition that will have the treshhold to each actvatioin fucntion (Without activation is equal 0, sigmoide is equal 0.5)
        a = node.a
        psi = self._calcule_psi(row,a)
        if (psi < threshold):
            node = node.left
        else:  node = node.right
        return node 
    
    "Classificar os Dados"
    def predict(self,x_teste, threshold):
        Predict = []
        for x in x_teste:
             label = self.__find_label(x,threshold)
             Predict.append(label)
        return Predict

   
       
    def __Eval_bin_or_multi(self,y_size,Predict, y_test):
        if y_size == 2:
           return self.__Evaluate_Metrics_Binary(Predict,y_test)
        elif y_size > 2:
            return self.__Evaluate_metrics_multi_class(Predict, y_test)
        
    def __viz_confusion(self,Predict, real):
        cm = confusion_matrix(real, Predict)
        cmd = ConfusionMatrixDisplay(cm)
        cmd.plot(cmap=plt.cm.Blues)
        cmd.ax_.set(xlabel='Predicted', ylabel='True')  
        #st.pyplot()
        return cmd
    


class P2(EntropyClassifier):
         
        
      def _Line(self,X,a):
          super()._Line
          if( self.activation == 'heaviside'):
             #D_neg, D_pos = self._gerar_reta_p2(X,a)
             #f_pos, f_neg =  self._frequency_heaviside(D_neg, D_pos)
            f_pos, f_neg, D_pos, D_neg, S_a_pos, S_a_neg,S_a_total =  self._total_entropy_heavi(X,a) 
          return f_pos, f_neg, D_pos, D_neg, S_a_pos, S_a_neg,S_a_total
         
    

            
      "Return vector a rand nums #a = (#X atributes)+1"
      def _random_a(self,X):
            super()._random_a #cardinal de a é definido por 3n
            aa = []
            X_lenght = X.shape[1]-1
            X_lenght = X_lenght + int((X_lenght*((X_lenght+1)/2))+1)
            return self._normalization((np.random.uniform(-1/2,1/2, ((X_lenght,1)))).T)
             #return self._normalization((np.random.uniform(-1/2,1/2,((X.shape[1],1)))).T) 
       

        
      def _gerar_reta_p2(self,X,a):
            z = X.shape[1] #Limite onde meu a vai multiplicar o vetor X de atributos
            psi = np.add(a[0,0],np.dot(a[0,1:z], X[:,0:-1].T)) #a0+a1*x1+a2*x2 ELE PEGA ATE 1 ANTES DO Z POR ISSO NAO PRECISO FAZER Z++
            psi = np.add(psi ,np.dot(a[0,z:z*2-1],  np.square(X[:,0:-1].T)))
            zz = z*2-1 #(zz*2 pq agora eu multiplico pelos atributos qo quadrado logo vai ate o dobro dos coeficientes a anteriores, e -1 pq X aqui vem com a cl)
            c, d = np.triu_indices(X.shape[1]-1, 1)
            psi = np.add(psi,np.sum(np.multiply( np.multiply(a[0,zz:], X[:,c]), X[:,d]), axis=1))
            id_neg = np.nonzero(psi < 0)
            id_pos = np.nonzero(psi >= 0)
            return X[id_neg], X[id_pos]  
      
        
        
      
      def _total_entropy_heavi(self,X,a):
          """
          Input: f_pos -> positive frequencies of each class in a dict
                 f_neg -> negative frequencies of each class in a dict
                 D_pos -> Set of all events presents in negative demiplan
                 D_neg -> Set of all events presents in positive demiplan
                 
          Output: S_a_pos -> Scalar positive Entropy
                  S_a_neg -> Scalar negative Entropy
                  S_a_Total -> Scalar of total Entropy
          """
          super()._total_entropy_heavi(X, a)
          D_neg, D_pos = self._gerar_reta_p2(X,a)
          f_pos, f_neg =  self._frequency_heaviside(D_neg, D_pos)
          S_a_pos, S_a_neg = self._parcial_entropy(f_pos, f_neg)
          S_a_total = ((len(D_neg)* S_a_neg) + (len(D_pos)*S_a_pos))/len(X)
          return f_pos, f_neg, D_pos, D_neg, S_a_pos, S_a_neg,S_a_total  


      def _calcule_psi(self,row,aç):
            super()._calcule_psi
            z = row.size+1 
            j = 0
            psi = np.add(aç[0,0],np.dot(aç[0,1:z], row.T))
            psi = np.add(psi ,np.dot(aç[0,z:z*2-1],  np.square(row.T)))
            zz = z*2-1
            c, d = np.triu_indices(row.size, 1)
            psi = np.add(psi,np.sum(np.multiply( np.multiply(aç[0,zz:], row[c]), row[d])))
            return psi
        

