from collections import deque
import functools as fn
"""
1 - A Binary Tree Structre
2 - Each Node have a index, left and right child (can be empty), and a parent (if is root dont have pany parent)

"""

class BinaryTreeNode:
    """
    index (INT) --> the index of each node, it is based on it that we will find each node
    left (OBJECT) --> the node left child, wich is a another node
    right (OBJECT) --> '''''''' right ''''''''''''''''''''''''''''
    children (LIST) --> list containing the children of a given node
    
    
    print_Tree(self) --> call printBTree a Function to print the structure of the tree
    """
    def __init__(self, index, left=None, right=None, children = [], parent = [], a = [], C_k= []):
        self.index = index
        self.left = left
        self.right = right
        self.children = []
        self.parent = []
        self.a = []
        self.data = {}
        self.C_k = []
        
    """ A diferenca entre __str__ and __repr__ is that str return a human redeable version of the object, while
    repr return """
    
    def __str__(self):
        return f"BinaryTreeNode(index={self.index}, left={self.left}, right={self.right}, a={self.a})"     
    
    def __repr__(self):
        return f"BinaryTreeNode(index={self.index}, left={self.left!r}, right={self.right!r}, children={self.children!r}, parent={self.parent!r})"
   
    def __eq__(self, other):  #eq retorna se um index é igual a outro , é um operador de comparacao em outras palavras.
        if self.index == other.index and \
            self.right == other.right and \
                self.left == other.left:
            return True
        else:
            return False
        
    def get_left(self):
        return self.left
    
    #def __str__(self, level=0):
     #   ret = "\t"*level+repr(self.value)+"\n"
      #  for child in self.children:
       #     ret += child.__str__(level+1)
       # return ret
    
    def printTree(self):
        printBTree(self,lambda n:(str(n.index),n.left,n.right))  
       
class BinaryTree:
    
    
    """
    
    """
    def __init__(self, root_index=None):
        # maps from BinaryTreeNode index to BinaryTreeNode instance.
        # Thus, BinaryTreeNode indexes must be unique.
        self.nodes = {} #a dict is created to store each node in the icorresponding index
        if root_index is not None: #Verify if root_index is not none
            # create a root BinaryTreeNode
            self.root = BinaryTreeNode(root_index) #Atribuo a o root meu primeiro node
            self.nodes[root_index] = self.root  #coloco no dicionario nodes mru primeiro node
            
    def get_Node (self, index):
        return 
   
    def add_Node(self, index, left_index=None, right_index=None):
        if index not in self.nodes: # Se o valor que eu colocar nao tiver no dict de nodes ele cria pra mim
            # BinaryTreeNode with given index does not exist, create it
            self.nodes[index] = BinaryTreeNode(index) #Se o node que estou adicionando pra ser pai nao estiver ainda declarado, eu declaro aqui e adiciono a nodes
        # invariant: self.nodes[index] exists

        # handle left child
        if left_index is None: #Se o left index for none o node left recebe none
            self.nodes[index].left = None
            self.nodes[index].children.append(None)  
        else:
            if left_index not in self.nodes:  # Se o left estiver com algum index ele recebe este valor
                self.nodes[left_index] = BinaryTreeNode(left_index) #Aqui to adicionando o left para a lista de nodes
                self.nodes[left_index].parent.extend((self.nodes[index],0)) #Coloco os dados na lista parent
            # invariant: self.nodes[left_index] exists
            self.nodes[index].left = self.nodes[left_index]
            self.nodes[index].children.append(self.nodes[index].left)    
        # handle right child
        if right_index == None:
            self.nodes[index].right = None
            self.nodes[index].children.append(None)  
        else:
            if right_index not in self.nodes:
                self.nodes[right_index] = BinaryTreeNode(right_index)
                self.nodes[right_index].parent.extend((self.nodes[index],1))
            # invariant: self.nodes[right_index] exists
            self.nodes[index].right = self.nodes[right_index]
            self.nodes[index].children.append(self.nodes[index].right)
      #Esta função somente apaga o index passado (nao apagarecursivamente os filhos e parentes do node pai)       
    def del_Node(self, index):
        if index not in self.nodes:
            raise ValueError('%s not in tree' % index)
        # remove index from the list of nodes
        #self.nodes[index].parents
        for i in self.nodes:
            if self.nodes[index] in self.nodes[i].children: #Verify if the deleted node is children for another node, and if is a children, the 2 children are vanished and the array return a empty list
                self.nodes[i].children = []
        del self.nodes[index]
        # if node removed is left/right child, update parent node
        for k in self.nodes:
            if self.nodes[k].left and self.nodes[k].left.index == index:
                self.nodes[k].left = None
                self.nodes[k].right = None
            if self.nodes[k].right and self.nodes[k].right.index == index:
                self.nodes[k].right = None
                self.nodes[k].left = None
        return True
#Esta é a versão que apaga tudo recursivamente
    def del_Node2(self, index):
        if index not in self.nodes:
            raise ValueError('%s not in tree' % index)
        # remove index from the list of nodes
        #self.nodes[index].parents
        rootdel = self.nodes[index].parent[0].index #root del recebe o index do pai do node que quero excluir
        del_nodes = self.traverse_postorder(self.nodes[rootdel]) #Lista dos indexes dos nodes subsequentes de rootdel
        self.nodes[rootdel].children = [] #Ja apago a lista de filhos 
        for i in del_nodes[0:-1]: #Apago cada node cm o index igual aos filhos e subfilhos do rootdel
            del self.nodes[i]
        # if node removed is left/right child, update parent node
        for k in self.nodes:
            if self.nodes[k].left and self.nodes[k].left.index == index:
                self.nodes[k].left = None
                self.nodes[k].right = None
            if self.nodes[k].right and self.nodes[k].right.index == index:
                self.nodes[k].right = None
                self.nodes[k].left = None
        return True
        
    def _height(self, node):
        if node is None:
            return 0
        else:
            return 1 + max(self._height(node.left), self._height(node.right))

    def height(self):
        return self._height(self.root)

    def size(self):
        return len(self.nodes)

    def __repr__(self):
        return str(self.traverse_inorder(self.root))


    # visit left child, root, then right child
    def traverse_inorder(self, node, reachable=None):
        if not node or node.index not in self.nodes:
            return
        if reachable is None:
            reachable = []
        self.traverse_inorder(node.left, reachable)
        reachable.append(node.index)
        self.traverse_inorder(node.right, reachable)
        return reachable

    # visit left and right children, then root
    # root of tree is always last to be visited
    def traverse_postorder(self, node, reachable=None):
        if not node or node.index not in self.nodes: #Verifica se o Node esta na lista, neste caso recursivo a cada iteracao é fornecido o filho do node atual para a funcao, e se nao tiver filho ela retorna nada ou seja sai da funcao
            return 
        if reachable is None:
            reachable = []
        self.traverse_postorder(node.left, reachable) #Passo o node left, e a lista que agora nao é None e sim vazia
        self.traverse_postorder(node.right, reachable) #Neste caso quando nao da return em nada ele vem para esta lista aqui
        reachable.append(node.index) 
        return reachable
    
    def child(self, index, return_node = False):        
        if index not in self.nodes:
            return "This Index dont exist"
        if return_node == True: #Return a list with 2 children nodes
           return  self.nodes[index].children
        if return_node == False:
           left_child = self.nodes[index].children[0].index
           right_child = self.nodes[index].children[1].index
           return left_child,right_child
    
    def parent(self, index, return_node = False):        
        if index not in self.nodes:
            return "This Index dont exist"
        if return_node == True: #Return a list with 2 children nodes
           return  self.nodes[index].parent
        if return_node == False:
           parent_node = self.nodes[index].parent[0].index
           parent_side = self.nodes[index].parent[1]
           return parent_node,parent_side
       
    def is_Leaf(self, index):
        if not self.nodes[index].children:
            return 1 # 1 é folha
        else:
            return 0 # 0 é node
    
    def Le (self):
        leaves = []
        for k in self.nodes:
            if self.is_Leaf(k) == True:
               leaves.append(k)
        return leaves
            
    def In (self):
        nodes = []
        for k in self.nodes:
            if self.is_Leaf(k) == False:
               nodes.append(k)
        return nodes   
    

    def printTreer(self):
        self.root.printTree()


def printBTree(node, nodeInfo=None, inverted=False, isTop=True):

       # node value string and sub nodes
       stringValue, leftNode, rightNode = nodeInfo(node)
    
       stringValueWidth  = len(stringValue)
    
       # recurse to sub nodes to obtain line blocks on left and right
       leftTextBlock     = [] if not leftNode else printBTree(leftNode,nodeInfo,inverted,False)
    
       rightTextBlock    = [] if not rightNode else printBTree(rightNode,nodeInfo,inverted,False)
    
       # count common and maximum number of sub node lines
       commonLines       = min(len(leftTextBlock),len(rightTextBlock))
       subLevelLines     = max(len(rightTextBlock),len(leftTextBlock))
    
       # extend lines on shallower side to get same number of lines on both sides
       leftSubLines      = leftTextBlock  + [""] *  (subLevelLines - len(leftTextBlock))
       rightSubLines     = rightTextBlock + [""] *  (subLevelLines - len(rightTextBlock))
    
       # compute location of value or link bar for all left and right sub nodes
       #   * left node's value ends at line's width
       #   * right node's value starts after initial spaces
       leftLineWidths    = [ len(line) for line in leftSubLines  ]                            
       rightLineIndents  = [ len(line)-len(line.lstrip(" ")) for line in rightSubLines ]
    
       # top line value locations, will be used to determine position of current node & link bars
       firstLeftWidth    = (leftLineWidths   + [0])[0]  
       firstRightIndent  = (rightLineIndents + [0])[0] 
    
       # width of sub node link under node value (i.e. with slashes if any)
       # aims to center link bars under the value if value is wide enough
       # 
       # ValueLine:    v     vv    vvvvvv   vvvvv
       # LinkLine:    / \   /  \    /  \     / \ 
       #
       linkSpacing       = min(stringValueWidth, 2 - stringValueWidth % 2)
       leftLinkBar       = 1 if leftNode  else 0
       rightLinkBar      = 1 if rightNode else 0
       minLinkWidth      = leftLinkBar + linkSpacing + rightLinkBar
       valueOffset       = (stringValueWidth - linkSpacing) // 2
    
       # find optimal position for right side top node
       #   * must allow room for link bars above and between left and right top nodes
       #   * must not overlap lower level nodes on any given line (allow gap of minSpacing)
       #   * can be offset to the left if lower subNodes of right node 
       #     have no overlap with subNodes of left node                                                                                                                                 
       minSpacing        = 2
       rightNodePosition = fn.reduce(lambda r,i: max(r,i[0] + minSpacing + firstRightIndent - i[1]), \
                                     zip(leftLineWidths,rightLineIndents[0:commonLines]), \
                                     firstLeftWidth + minLinkWidth)
    
       # extend basic link bars (slashes) with underlines to reach left and right
       # top nodes.  
       #
       #        vvvvv
       #       __/ \__
       #      L       R
       #
       linkExtraWidth    = max(0, rightNodePosition - firstLeftWidth - minLinkWidth )
       rightLinkExtra    = linkExtraWidth // 2
       leftLinkExtra     = linkExtraWidth - rightLinkExtra
    
       # build value line taking into account left indent and link bar extension (on left side)
       valueIndent       = max(0, firstLeftWidth + leftLinkExtra + leftLinkBar - valueOffset)
       valueLine         = " " * max(0,valueIndent) + stringValue
       slash             = "\\" if inverted else  "/"
       backslash         = "/" if inverted else  "\\"
       uLine             = "¯" if inverted else  "_"
    
       # build left side of link line
       leftLink          = "" if not leftNode else ( " " * firstLeftWidth + uLine * leftLinkExtra + slash)
    
       # build right side of link line (includes blank spaces under top node value) 
       rightLinkOffset   = linkSpacing + valueOffset * (1 - leftLinkBar)                      
       rightLink         = "" if not rightNode else ( " " * rightLinkOffset + backslash + uLine * rightLinkExtra )
    
       # full link line (will be empty if there are no sub nodes)                                                                                                    
       linkLine          = leftLink + rightLink
    
       # will need to offset left side lines if right side sub nodes extend beyond left margin
       # can happen if left subtree is shorter (in height) than right side subtree                                                
       leftIndentWidth   = max(0,firstRightIndent - rightNodePosition) 
       leftIndent        = " " * leftIndentWidth
       indentedLeftLines = [ (leftIndent if line else "") + line for line in leftSubLines ]
    
       # compute distance between left and right sublines based on their value position
       # can be negative if leading spaces need to be removed from right side
       mergeOffsets      = [ len(line) for line in indentedLeftLines ]
       mergeOffsets      = [ leftIndentWidth + rightNodePosition - firstRightIndent - w for w in mergeOffsets ]
       mergeOffsets      = [ p if rightSubLines[i] else 0 for i,p in enumerate(mergeOffsets) ]
    
       # combine left and right lines using computed offsets
       #   * indented left sub lines
       #   * spaces between left and right lines
       #   * right sub line with extra leading blanks removed.
       mergedSubLines    = zip(range(len(mergeOffsets)), mergeOffsets, indentedLeftLines)
       mergedSubLines    = [ (i,p,line + (" " * max(0,p)) )       for i,p,line in mergedSubLines ]
       mergedSubLines    = [ line + rightSubLines[i][max(0,-p):]  for i,p,line in mergedSubLines ]                        
    
       # Assemble final result combining
       #  * node value string
       #  * link line (if any)
       #  * merged lines from left and right sub trees (if any)
       treeLines = [leftIndent + valueLine] + ( [] if not linkLine else [leftIndent + linkLine] ) + mergedSubLines
    
       # invert final result if requested
       treeLines = reversed(treeLines) if inverted and isTop else treeLines
    
       # return intermediate tree lines or print final result
       if isTop : print("\n".join(treeLines))
       else     : return treeLines                  

#O que eu vou fazer é pegar o parente do que eu quero excluir colocar ele no postorder ai excluo todo resto




Tree = BinaryTree(1)
#IF I CALL tree alone will print the elements on th tree using inorder_traverse
Tree.add_Node(1,3,4)
Tree.add_Node(4,5,6)
Tree.add_Node(3,8,9)
Tree.add_Node(9,10,11)
Tree.add_Node(8,12,13)
Tree.add_Node(10,14,15)
Tree.add_Node(15,16,17)
root = Tree.root
Tree.printTreer()
noder = Tree.nodes[8] #ESSE é o meu get
print(str(noder))

import numpy as np

#Inner Nodes Arr
Tree.In()

#Leaf Nodes Arr
Tree.Le()

#Children nodes of index 8 (or node 8)
Tree.child(8)

#Parent nodes of index 8 (the first item is the parent, and second is 0 if its left side or 1 if its right side)
Tree.parent(8)
#Return 1 if a node is leaf and 0 otherwise
Tree.is_Leaf(8)
Tree.printTreer()
Tree.is_Leaf(root.left.index)
root.left.index
root.data
keys =  ["data", "entropy", "classe"]
values = [[1,5,6,8,9,7,4,5],100,5]
root.data = {k: v for k, v in zip (keys, values)}
print(root.data)
root.data["classe"]
root = Tree.nodes[15]
root = root.left
gg = Tree.nodes[1]
Tree.is_Leaf(gg.index)
root.C_k.append (np.array([-0.999131, 0.87589, 0]))
"""

In this example, we defined the __repr__ method to return a string that represents the object in a way that can be
 used to recreate the object. When we call the repr() function on the object, Python calls the __repr__ method and
 returns the string that it returns.

Note that the __repr__ method is intended to be used by developers for debugging and development purposes, while 
the __str__ method is intended to be used by users to obtain a human-readable string representation of the object.



print("dd")
print(Tree)
#lista = teste.nodes[3]
#lista2 = teste.nodes[3].children[1]
root = Tree.root
root.printTreer()
Tree.del_Node2(9)
root.printTreer()
Tree.child(10)
print()
criancas = Tree.child(9)
#teste.traverse_postorder(teste.nodes[9])
pais = Tree.parent(9)
Tree.is_Leaf(10)
Tree.Le()
Tree.In()
Tree.height()
folha = Tree.nodes[1]
print(folha.left)
#kct = teste.traverse_postorder(lista)
#teste.traverse_inorder(lista)
#dictnode= teste.nodes


#teste.nodes.pop(9,8)
#lista == teste.nodes[3].right
#teste.del_Node(9)
#root = teste.root
#root.printTreer()
#hhh = teste.root.children
#dictt.clear()

#lista = root[9]
#hhh.pop()
"""




"""
teste = BinaryTree(1)
teste.add(1,3,4)
teste.add(4,5,6)

print(teste)
printHeapTree([1, 3, 4, 5, 6])
teste.nodes["nodes"]
teste = BinaryTree(1)
teste.add(1,3,4)
teste.add(3,5,6)
teste.height()
print(teste[0])
ggg = repr(teste)
type(teste)
for i in teste:
    print(i)
hhh = teste.nodes
teste.nodes
print(hhh)
"""


 # Output: 42
